import librosa
import torch
from transformers import ClapProcessor, ClapModel
import numpy as np
from typing import List, Dict
from openai import OpenAI
import os
import IPython
from multimodal_lancedb import MusicDatabase

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
    def __init__(self, db_path: str = "./.lancedb", music_dir: str = "music"):
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
    
    def search_music(self, query: str, top_k: int = 5) -> str:
        """
        Search for music using both audio and text embeddings
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            str: LLM-generated explanation of recommendations
        """
        # Get query embeddings
        clap_vector = self.embedding_processor.get_text_embedding(query, use_clap=True)
        openai_vector = self.embedding_processor.get_text_embedding(query, use_clap=False)
        
        # Search in both tables
        audio_results = self.db.search_songs(clap_vector, self.db.tables["audio"], top_k)
        text_results = self.db.search_songs(openai_vector, self.db.tables["text"], top_k)
        
        # Print individual search results
        print("Audio Search Results:")
        for _, row in audio_results.iterrows():
            print(f"- {row['song_name']}")
    
        print("\nText Search Results:")
        for _, row in text_results.iterrows():
            print(f"- {row['song_name']} by {row['artist']}")

        # Find overlapping recommendations
        common_songs = set(audio_results['song_name']) & set(text_results['song_name'])
        final_recommendations = text_results[text_results['song_name'].isin(common_songs)]
        
        if len(final_recommendations) == 0:
            return "No matching songs found in both audio and text searches."
        
         # Get audio paths for final recommendations
        audio_paths = []
        for _, row in final_recommendations.iterrows():
            audio_path = self.db.generate_audio_path(row['artist'], row['song_name'])
            if os.path.exists(audio_path):
                audio_paths.append(audio_path)

        # Format recommendations for LLM
        recommendations = []
        for _, row in final_recommendations.iterrows():
            recommendation = {
                "song_name": row["song_name"],
                "artist": row["artist"],
                "mood": row["mood"],
                "video_theme": row["video_theme"],
                "genre": row["genre"],
                "instrument": row["instrument"],
                "bpm": row["bpm"],
                "description": row["lmm_description"],
                "similarity_score": float(row["_distance"])
            }
            recommendations.append(recommendation)
        
        # Generate explanation using LLM
        explanation = self._generate_explanation(query, recommendations)

        return {
        "audio_results": audio_results['song_name'].tolist(),
        "text_results": text_results['song_name'].tolist(),
        "final_results": final_recommendations[['song_name', 'artist']].to_dict('records'),
        "explanation": explanation,
        "audio_paths": audio_paths
        }
    
    def _generate_explanation(self, query: str, recommendations: List[Dict]) -> str:
        """Generate LLM explanation for recommendations"""
        prompt = f"""As a professional music recommendation assistant, please generate a natural and detailed recommendation description based on the user's preferences and the characteristics of the recommended songs.

User needs: {query}

Recommended songs:
"""
        for i, rec in enumerate(recommendations, 1):
            similarity_percentage = (1 - rec["similarity_score"]) * 100
            prompt += f"""
{i}. {rec['song_name']} - {rec['artist']}
   Genre: {rec['genre']}
   Mood: {rec['mood']}
   Video Theme: {rec['video_theme']}
   Instrument: {rec['instrument']}
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
            temperature=0.7
        )

        return response.choices[0].message.content
