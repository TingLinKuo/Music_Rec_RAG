import lancedb
import pyarrow as pa
import os
import pandas as pd
from typing import List, Dict

def combine_info(music_df):
    """
    Combines Tags and LMM Desciption into a single string for each row in the DataFrame.
    
    Returns:
    pd.DataFrame: The updated DataFrame with a new 'combined_info' column.
    """
    # Let nan as empty string
    target_columns = ["Mood", "Genre", "Instrument", "Video Theme", "Others"]

    for col in target_columns:
        music_df[col] = music_df[col].apply(
        lambda x: "" if pd.isna(x) or str(x).strip().lower() in ["nan", "none"] else str(x).strip()
    )
    
    # Drop useless columns
    music_df = music_df.drop(columns=["Useless"], errors='ignore')

    music_df['combined_info'] = music_df.apply(
        lambda row: ( 
            f"Moods: {row['Mood']}. "
            f"Video Themes: {row['Video Theme']}. "
            f"Instruments: {row['Instrument']}. "
            f"Genres: {row['Genre']}. "
            f"Other tags: {row['Others']}. "
            f"Description: {row['LMM Description']}"
        ),
        axis=1
    )

    return music_df

class MusicDatabase:
    def __init__(self, db_path: str = "./.lancedb", music_dir: str = "music"):
        """Initialize MusicDatabase with LanceDB connection"""
        self.db = lancedb.connect(db_path)
        self.music_dir = music_dir
        self.tables = {
            "audio": "music_audio",
            "text": "music_text"
        }
    
    def import_text_descriptions_from_csv(self, csv_path: str, embedding_processor) -> List[Dict]:
        """
        Import song descriptions from a CSV file and add them to the database
        This function will also do llm_description embedding
        
        Args:
            csv_path (str): Path to the CSV file
            embedding_processor: Instance of EmbeddingProcessor to generate embeddings
            
        Returns:
            List[Dict]: List of processed records with audio paths
        """
        
        try:
            df = pd.read_csv(csv_path)
            df = combine_info(df)
            required_columns = [ 
                "source", "song_name", "artist", 
                "Classification", 
                # tags
                "Mood", "Video Theme", "Instrument", "Genre", "BPM", "Others",
                # description
                "LMM Description", 
                # combined info from combine_info function
                "combined_info"]

            # Verify CSV structure, check for required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
            # VER type to str
            exclude_column = "BPM"
            for column in df.columns:
                if column != exclude_column:
                    df[column] = df[column].astype(str)

            # Process each row
            text_records = []
            audio_records = []
            
            for _, row in df.iterrows():
                # Generate text embedding
                text_vector = embedding_processor.get_text_embedding(row['combined_info'], use_clap=False)
                text_vector = [float(x) for x in text_vector]              
                
                # generate text table record, this is schema for table in lancedb
                record = {
                    "source": row["source"],
                    "song_name": row["song_name"],
                    "artist": row["artist"],
                    "mood": row["Mood"],
                    "video_theme": row["Video Theme"],
                    "instrument": row["Instrument"],
                    "genre": row["Genre"],
                    "other_tags": row["Others"],
                    "bpm": int(row["BPM"]) if pd.notna(row["BPM"]) and row["BPM"] != "" else None,
                    "lmm_description": row["LMM Description"],
                    "combined_info": row["combined_info"],
                    "text_vector": text_vector
                }
                text_records.append(record)
                
                # Generate audio paths
                audio_path = self.generate_audio_path(row['artist'], row['song_name'])
                # Add to audio records list
                audio_records.append({
                    "song_name": row["song_name"],
                    "path": audio_path
                })

            # Add to database
            table = self.db.open_table(self.tables["text"])
            table.add(text_records)
            print(f"Successfully imported {len(text_records)} records from {csv_path}")

            return audio_records
            
        except Exception as e:
            print(f"Error importing CSV: {str(e)}")
            raise

    def generate_audio_path(self, artist: str, song_name: str) -> str:
        # Generate the audio file path based on artist and song name
        base_name = f"{artist} - {song_name}"
        mp3_path = os.path.join(self.music_dir, base_name + ".mp3")
        wav_path = os.path.join(self.music_dir, base_name + ".wav")

        if os.path.exists(mp3_path):
            return mp3_path
        elif os.path.exists(wav_path):
            return wav_path
        else:
            raise FileNotFoundError(f"cannot find {base_name}.mp3 or .wav")
        
    
    def initialize_tables(self, vector_dims: Dict[str, int]):
        """Initialize database tables if they don't exist"""
        # Schema for audio table
        audio_schema = pa.schema([
            pa.field("song_name", pa.string()),
            pa.field("song_path", pa.string()),
            pa.field("audio_vector", pa.list_(pa.float32(), vector_dims['audio']))
        ])
        
        # Schema for text table
        text_schema = pa.schema([
            pa.field("source", pa.string()),
            pa.field("song_name", pa.string()),
            pa.field("artist", pa.string()),
            pa.field("mood", pa.string()),
            pa.field("video_theme", pa.string()),
            pa.field("genre", pa.string()),
            pa.field("instrument", pa.string()),
            pa.field("other_tags", pa.string()),
            pa.field("bpm", pa.int32()),
            pa.field("lmm_description", pa.string()),
            pa.field("combined_info", pa.string()),
            pa.field("text_vector", pa.list_(pa.float32(), vector_dims['text']))
        ])
        
        # Create tables if they don't exist
        if self.tables["audio"] not in self.db.table_names():
            self.db.create_table(self.tables["audio"], schema=audio_schema)
        
        if self.tables["text"] not in self.db.table_names():
            self.db.create_table(self.tables["text"], schema=text_schema)
    
    def add_song_audio(self, song_name: str, song_path: str, audio_vector: List[float]):
        """Add song audio embedding to database"""
        table = self.db.open_table(self.tables["audio"])
        table.add([{
            "song_name": song_name,
            "song_path": song_path,
            "audio_vector": audio_vector
        }])
    
    def add_song_text(self, song_data: Dict):
        """Add song text data and embedding to database"""
        table = self.db.open_table(self.tables["text"])
        table.add([song_data])
    
    def search_songs(self, query_vector: List[float], table_name: str, top_k: int,  metric: str = "cosine") -> pd.DataFrame:
        """
        Search for similar songs in specified table
        By default, l2 will be used as metric type. (歐基里德距離)
        LanceDb supports cosine and dot production as well.
        """
        assert metric in {"l2", "cosine", "dot"}, f"Unsupported metric: {metric}"
        table = self.db.open_table(table_name)
        return table.search(query_vector).metric(metric).limit(top_k).to_df()
    