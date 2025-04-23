import librosa
import torch
from transformers import ClapProcessor, ClapModel
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from openai import OpenAI
import os
from multimodal_lancedb import MusicDatabase
import lancedb

class EmbeddingProcessor:
    def __init__(self, clap_model_name: str = "laion/clap-htsat-unfused"):
        """
        Initialize embedding processors for both CLAP and OpenAI
        
        Args:
            clap_model_name (str): Name of the CLAP model to use
        """
        # Initialize CLAP
        self.clap_processor = ClapProcessor.from_pretrained(clap_model_name)
        self.clap_model = ClapModel.from_pretrained(clap_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clap_model = self.clap_model.to(self.device)
        
        # Initialize OpenAI
        self.llm = OpenAI()
    
    def get_audio_embedding(self, audio_path: str) -> np.ndarray:
        """Get audio embedding using CLAP"""
        audio_tensor = self._preprocess_audio(audio_path).to(self.device)
        audio_inputs = self.clap_processor(
            audios=audio_tensor.cpu().numpy(),
            sampling_rate=48000,
            return_tensors="pt"
        ).to(self.device)
        
        inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}
        
        with torch.no_grad():
            audio_embedding = self.clap_model.get_audio_features(**inputs)
            return audio_embedding.squeeze().cpu().numpy()

    def get_text_embedding(self, text: str, use_clap: bool = True) -> np.ndarray:
        # Get text embedding using either CLAP or OpenAI
        if use_clap:
            text_inputs = self.clap_processor(
                text=text,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            text_embedding = self.clap_model.get_text_features(**text_inputs)
            return text_embedding.detach().squeeze().cpu().numpy()
        else:
            response = self.llm.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding

    def _preprocess_audio(self, audio_path: str, max_duration: int = 30) -> torch.Tensor:
        # Load audio file and resample to 48kHz
        audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=max_duration)
        # Convert audio to a tensor
        return torch.tensor(audio).unsqueeze(0)

class MusicSearchSystem:
    def __init__(self, db_path: str = "./.lancedb_2", music_dir: str = "music"):
        """Initialize the Music Search System with a LanceDB connection"""
        self.embedding_processor = EmbeddingProcessor()
        self.db = MusicDatabase(db_path, music_dir)
        
        # Initialize database with vector dimensions
        self.db.initialize_tables({
            'audio': 512,  # CLAP embedding dimension
            'text': 1536   # OpenAI embedding dimension
        })
    
    def import_text_descriptions(self, csv_path: str, process_audio: bool = True):
        """Import text descriptions from a CSV file"""
        audio_records = self.db.import_text_descriptions_from_csv(csv_path, self.embedding_processor)

        if process_audio:
            self.add_audio_for_existing_songs(audio_records)
    
    def add_audio_for_existing_songs(self, audio_files: List[Dict[str, str]]):
        # Add audio files for songs that already have text descriptions
        for audio_file in audio_files:
            try:
                audio_vector = self.embedding_processor.get_audio_embedding(audio_file['path'])
                self.db.add_song_audio(
                    song_name=audio_file['song_name'],
                    song_path=audio_file['path'],
                    audio_vector=audio_vector.tolist()
                )
                print(f"Processed audio for: {audio_file['song_name']}")
            except Exception as e:
                print(f"Error processing audio for {audio_file['song_name']}: {str(e)}")
                continue
    
    def search_music(self, query: str, top_k: int = 200) -> str:
        """
        Search for music using both audio and text embeddings
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return of each embedding
            
        Returns:
            str: LLM-generated explanation of recommendations
        """
        ranker = Ranker(db=self.db, weight=(0.3, 0.7))
        recommendations = ranker.ranking_score_based(query, top_k)
        
        # # Get query embeddings
        # clap_vector = self.embedding_processor.get_text_embedding(query, use_clap=True)
        # openai_vector = self.embedding_processor.get_text_embedding(query, use_clap=False)
        
        # # Search in both tables
        # audio_results = self.db.search_songs(clap_vector, self.db.tables["audio"], top_k)
        # text_results = self.db.search_songs(openai_vector, self.db.tables["text"], top_k)
        
        # # Print individual search results
        # print("Audio Search Results:")
        # for _, row in audio_results.iterrows():
        #     print(f"- {row['song_name']}")
    
        # print("\nText Search Results:")
        # for _, row in text_results.iterrows():
        #     print(f"- {row['song_name']} by {row['artist']}")

        # # Find overlapping recommendations
        # common_songs = set(audio_results['song_name']) & set(text_results['song_name'])
        # final_recommendations = text_results[text_results['song_name'].isin(common_songs)]
        
        # if len(final_recommendations) == 0:
        #     return "No matching songs found in both audio and text searches."
        
        #  # Get audio paths for final recommendations
        # audio_paths = []
        # for _, row in final_recommendations.iterrows():
        #     audio_path = self.db.generate_audio_path(row['artist'], row['song_name'])
        #     if os.path.exists(audio_path):
        #         audio_paths.append(audio_path)

        # # Format recommendations for LLM
        # recommendations = []
        # for _, row in final_recommendations.iterrows():
        #     recommendation = {
        #         "song_name": row["song_name"],
        #         "artist": row["artist"],
        #         "mood": row["mood"],
        #         "video_theme": row["video_theme"],
        #         "genre": row["genre"],
        #         "instrument": row["instrument"],
        #         "bpm": row["bpm"],
        #         "description": row["lmm_description"],
        #         # This is from LanceDB return 
        #         "similarity_score": float(row["_distance"])
        #     }
        #     recommendations.append(recommendation)
        
        # Generate explanation using LLM
        # explanation = self._generate_explanation(query, recommendations)

        return {
            "final_results": recommendations,
            # "explanation": explanation
        }
        # return {
        # "audio_results": audio_results['song_name'].tolist(),
        # "text_results": text_results['song_name'].tolist(),
        # "final_results": final_recommendations[['song_name', 'artist']].to_dict('records'),
        # "explanation": explanation,
        # "audio_paths": audio_paths
        # }
    
    def _generate_explanation(self, query: str, recommendations: List[Dict]) -> str:
        """Generate LLM explanation for recommendations"""
        prompt = f"""As a professional music recommendation assistant, please generate a natural and detailed recommendation description based on the user's preferences and the characteristics of the recommended songs.

        User needs: {query}

        Recommended songs:
        """
        for i, rec in enumerate(recommendations, 1):
            # similarity percentage
            similarity_percentage = (1 - rec["similarity_score"]) * 100
            prompt += f"""
                {i}. {rec['song_name']} - {rec['artist']}
                Genre: {rec['genre']}
                Mood: {rec['mood']}
                Video Theme: {rec['video_theme']}
                Instrument: {rec['instrument']}
                Other tags: {rec['other_tags']}
                BPM: {rec['bpm']}
                Desciption: {rec['description']}
                Similarity: {similarity_percentage:.1f}%
        """

        prompt += """
        Based on the above information, please explain:
        1. Description of why these songs fit the user
        2. Reasons for recommending these songs
        3. Characteristics of these songs

        Explain in a clear, understandable way, in a natural and friendly tone."""

        response = self.embedding_processor.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional music recommendation assistant who is good at explaining music characteristics and reasons for recommendation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

class Ranker:
    def __init__(self, db, weight: Tuple[float, float] = (0.5, 0.5)):
        """
        weight: (audio_weight, text_weight), must sum to 1
        """
        assert sum(weight) == 1.0, "Weights must sum to 1"
        self.db = db
        self.embedding_processor = EmbeddingProcessor()
        # Initialize weights
        self.audio_weight = weight[0]
        self.text_weight = weight[1]
    
    # for Score-based (加權融合)
    def ranking_score_based(self, query: str, top_k: int = 200) -> List[Dict]:
        # Get query embeddings
        clap_vector = self.embedding_processor.get_text_embedding(query, use_clap=True)
        openai_vector = self.embedding_processor.get_text_embedding(query, use_clap=False)
        
        # Search in both tables
        audio_results = self.db.search_songs(clap_vector, self.db.tables["audio"], top_k)
        text_results = self.db.search_songs(openai_vector, self.db.tables["text"], top_k)

        # print(audio_results)
        # print(text_results)
        
        # Add source label
        audio_results = audio_results.copy()
        audio_results["source"] = "audio"
        text_results = text_results.copy()
        text_results["source"] = "text"

        # Merge results, keeping needed columns
        metadata_columns = ["artist", "lmm_description"]
        search_db = lancedb.connect("./.lancedb_2")
        text_df = search_db.open_table("music_text").to_pandas()
        audio_with_metadata = audio_results.merge(
            text_df[["song_name"] + metadata_columns], 
            on="song_name", 
            how="left"
        )
        combined = pd.concat([
            audio_with_metadata, text_results], ignore_index=True)

        # Normalize distances
        def normalize_column(df, source_name):
            mask = df["source"] == source_name
            distances = df.loc[mask, "_distance"]
            min_val = distances.min()
            max_val = distances.max()
            norm_col = f"{source_name}_normalized_distance"
            df.loc[mask, norm_col] = (distances - min_val) / (max_val - min_val)
            return df

        combined = normalize_column(combined, "audio")      
        combined = normalize_column(combined, "text")
        
        grouped = (
            combined.groupby("song_name")
            .agg({
                "_distance": "min",
                "artist": lambda x: next((val for val in x if pd.notna(val)), None),
                "genre": "first",
                "mood": "first",
                "video_theme": "first",
                "instrument": "first",
                "other_tags": "first",
                "bpm": "first",
                "lmm_description": lambda x: next((val for val in x if pd.notna(val)), None),
                "audio_normalized_distance": "min",
                "text_normalized_distance": "min",
                "source": lambda x: ",".join(sorted(set(x)))
            })
            .reset_index()
        )

        # Compute weighted score per song (加權階段)
        def compute_weighted_score(row):
            audio_sim = 1 - row["audio_normalized_distance"] if pd.notna(row["audio_normalized_distance"]) else None
            text_sim = 1 - row["text_normalized_distance"] if pd.notna(row["text_normalized_distance"]) else None
            w_audio = self.audio_weight if audio_sim is not None else 0
            w_text = self.text_weight if text_sim is not None else 0
            total = w_audio + w_text if (w_audio + w_text) > 0 else 1
            return ((audio_sim or 0) * w_audio + (text_sim or 0) * w_text) / total
                  
            # sources = group["source"].tolist()
            # distances = group["_distance"].tolist()
            # # this is for text and audio are found
            # if len(distances) == 2:
            #     return self.audio_weight * distances[sources.index("audio")] + \
            #            self.text_weight * distances[sources.index("text")]
            # elif sources[0] == "audio":
            #     return self.audio_weight * distances[0] + self.text_weight  # penalize missing text
            # else:
            #     return self.text_weight * distances[0] + self.audio_weight  # penalize missing audio

        grouped["score"] = grouped.apply(compute_weighted_score, axis=1)
        # grouped["score"] = grouped["song_name"].map(
        #     lambda name: compute_weighted_score(combined[combined["song_name"] == name])
        # )

        # Extract individual raw distances ===
        def get_distance(name: str, source: str):
            match = combined[(combined["song_name"] == name) & (combined["source"] == source)]
            return float(match["_distance"].iloc[0]) if not match.empty else None

        grouped["audio_distance"] = grouped["song_name"].map(lambda n: get_distance(n, "audio"))
        grouped["text_distance"] = grouped["song_name"].map(lambda n: get_distance(n, "text"))

        # Audio path
        def resolve_audio_path(row):
            path = self.db.generate_audio_path(row["artist"], row["song_name"])
            return path if os.path.exists(path) else None

        grouped["audio_path"] = grouped.apply(resolve_audio_path, axis=1)

        # Format final output for LLM explanation or downstream use
        final_recommendations = []
        for _, row in grouped.sort_values("score", ascending=False).head(top_k).iterrows():
            final_recommendations.append({
                "song_name": row["song_name"],
                "artist": row["artist"],
                #"mood": row["mood"],
                #"video_theme": row["video_theme"],
                #"genre": row["genre"],
                #"instrument": row["instrument"],
                #"other_tags": row["other_tags"],
                #"bpm": row["bpm"],
                "description": row["lmm_description"],
                # This is from LanceDB return
                "similarity_score": row["score"],
                "similarity_audio": 1 - row["audio_normalized_distance"],
                "similarity_text": 1 - row["text_normalized_distance"],
                "sorce": row["source"],
                "audio_path": row["audio_path"]
            })

        return final_recommendations

# evaluation function
class Evaluator:
    def __init__(self, gt: List[str], recs: List[str]):
        self.gt = gt
        self.recs = recs

    def evaluate(self, k: int) -> Dict[str, float]:
        """Evaluate the recommendations using various metrics"""
        precision = precision_at_k(self.gt, self.recs, k)
        recall = recall_at_k(self.gt, self.recs, k)
        ndcg = ndcg_at_k(self.gt, self.recs, k)
        
        return {
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg
        }
    
# Evaluation metrics
def precision_at_k(gt, recs, k):
    recs_at_k = recs[:k]
    relevant = [r for r in recs_at_k if r in gt]
    return len(relevant) / k

def recall_at_k(gt, recs, k):
    recs_at_k = recs[:k]
    relevant = [r for r in recs_at_k if r in gt]
    return len(relevant) / len(gt) if len(gt) > 0 else 0.0

def ndcg_at_k(gt, recs, k):
    recs_at_k = recs[:k]
    dcg = 0.0
    for i, rec in enumerate(recs_at_k):
        if rec in gt:
            dcg += 1 / np.log2(i + 2)  # log2(i+2) since i starts from 0
    ideal_rels = [1] * min(len(gt), k)
    idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels)])
    return dcg / idcg if idcg > 0 else 0.0
