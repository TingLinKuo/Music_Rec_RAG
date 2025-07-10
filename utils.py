import librosa
import torch
from transformers import ClapProcessor, ClapModel
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from openai import OpenAI
import os
from multimodal_lancedb import MusicDatabase
from lancedb.rerankers import CohereReranker
import lancedb
import pyarrow as pa
import random
from google import genai
from google.genai import types

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
        self.google_api_key = os.getenv("GEMINI_API_KEY")
    
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
    
    def search_music(self, query: str, top_k: int = 200, rerank: bool = True, use_top_n_context: int = 1) -> Dict:
        """
        Search for music using both audio and text embeddings
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return of each embedding
            use_top_n_context (int): Number of top results to include in context for explanation
        
        Returns:
            str: LLM-generated explanation of recommendations
        """
        ranker = Ranker(db=self.db, weight=(0.3, 0.7))
        recommendations, recommendations_rerank = ranker.ranking_score_based(query, top_k)
        # recommendations = ranker.ranking_text_then_audio_rerank(query, top_k)
        final_recs = recommendations_rerank if rerank else recommendations
        explanation_context = [
        {
            "song_name": rec["song_name"],
            "artist": rec["artist"],
            "combined_info": rec["combined_info"]
        }
        for rec in final_recs[:use_top_n_context]
        ]
        choice, recommendation_block, prompt, explanation = self._generate_explanation(query, explanation_context)
        
        return {
            "choice": choice,
            "final_results": final_recs,
            "retrieval_context": recommendation_block,
            "explanation_prompt": prompt,
            "explanation": explanation
        }
        # return {
        # "audio_results": audio_results['song_name'].tolist(),
        # "text_results": text_results['song_name'].tolist(),
        # "final_results": final_recommendations[['song_name', 'artist']].to_dict('records'),
        # "explanation": explanation,
        # "audio_paths": audio_paths
        # }
    
    def _generate_explanation(self, query: str, recommendations: List[Dict]) -> str:
        """
        Generate explanation from LLM based on user query and the retrieved songs.
        This version uses combined_info and supports 1~3 context variations.
        """

        fake = """Assaf Ayalon - My Rhapsody Sounds - Short Version A
        Information: Moods: Uplifting, Happy, Carefree, Love, Playful. Video Themes: Business, Food, Education, Lifestyle, Urban. Instruments: Acoustic Guitar, Keys. Genres: Cinematic, Acoustic, Pop, Folk, Children, Corporate. Other tags: . Description: A positive and uplifting acoustic folk track with an upbeat rhythm and a happy melody. The acoustic guitar strums a catchy rhythm while the piano and celeste play a beautiful melody. The bass and drums add a lively beat to the track. This music is perfect for use in commercials, advertising, and other media projects that need a cheerful and optimistic mood.
        """
        fake_2 = '''Swirling Ship - Fixed - Short Version B
        Information: Moods: Serious, Dramatic, Scary, Dark. Video Themes: Time-Lapse, Drone Shots, Nature, Slow Motion. Instruments: Electric Guitar, Synth, Electronic Drums, Pads. Genres: Ambient, Country, Cinematic. Other tags: . Description: The music is mysterious and dramatic, featuring a soothing flute melody, evocative strings, and atmospheric pads. The mood is suspenseful and ominous, creating a sense of tension and intrigue. The instruments include the flute, strings, and pads. This music would be suitable for a wide range of video themes, including crime and mystery, horror, and suspense. It could also be used in documentaries, video games, and trailers to create a sense of tension and anticipation.
        '''    
        fake_3 = """The Mind Sweepers - Laid Back - Short Version A
        Information: Moods: Powerful, Serious, Angry. Video Themes: Road Trip, Sport & Fitness, Fashion, Industry. Instruments: Electric, Guitar, Acoustic Drums. Genres: Rock. Other tags: . Description: This is a powerful and energetic rock music track with catchy electric guitar riffs, hard hitting drums, and upbeat bass. The track is perfect for use in sports videos, advertising, commercials, corporate videos, and more. It will certainly add a touch of energy and excitement to your project."""

        recommendation_block = ""

        for i, rec in enumerate(recommendations, 1):
            recommendation_block += f"{i}. {rec['artist']} - {rec['song_name']}\n"
            recommendation_block += f"Information: {rec['combined_info']}\n\n"
        
        prompt = f"""You are a professional music recommendation assistant.
        Based on the following user need and the details of recommended songs, please generate a natural, clear, and engaging explanation.
        
        User needs: {query}

        Recommended songs:
        {recommendation_block}

        Based on the user's need and the details of the recommended songs, please provide a concise explanation that addresses the following:

        1. Why these songs match the user's need — or, if they don't match well, explain why.
        2. The reasons for selecting them - or, if inappropriate, highlight any mismatches or uncertainty.
        3. Their overall musical characteristics

        If the user's query is vague, confusing, or difficult to interpret, you may acknowledge the ambiguity and state that a strong connection cannot be determined.
        If the user's query does not appear to relate to music or song preferences at all, you should clearly state that no musical connection can be reasonably inferred.
        Be honest and objective. Do not invent or exaggerate connections. It is acceptable to express limitations or doubts.


        Keep the explanation under 100 words and no more than 5 sentences. Make it clear, specific, and easy for users to understand the connection between their need and the recommended music. Avoid redundancy."""
        # and either attempt a reasonable interpretation or 
        # Define system prompt
        GENERATION_SYS_PROMPT = """You are a professional music recommendation assistant who is good at explaining music characteristics and reasons for recommendation."""
        
        # Choose the LLM to use for explanation generation
        choice = ["openai", "gemini"][0] 
        if choice == "openai":
            # Use OpenAI to generate the explanation
            response = self.embedding_processor.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": GENERATION_SYS_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return_content = response.choices[0].message.content

        if choice == "gemini":
        # Use Google Gemini for explanation
            client = genai.Client(api_key=self.embedding_processor.google_api_key)
            response = client.models.generate_content(
                model="gemini-1.5-pro",
                config=types.GenerateContentConfig(
                    system_instruction=GENERATION_SYS_PROMPT,
                    temperature=0.3
                ),
                contents=prompt
            )

            return_content = response.text

        return choice, recommendation_block, prompt, return_content

    def rewrite_query_from_history(self, user_input: str, memory):
        history = memory.load_memory_variables({})["history"] # memory is from langchain
        history_str = "\n".join(f"{msg.type}: {msg.content}" for msg in history)

        system_prompt = (
        "You are a helpful assistant that rewrites user queries based on conversation history. "
        "Your job is to infer the user's full intent and produce a clear and complete music-related query."
        )

        user_prompt = f"""
        Here is the conversation history:
        {history_str}

        The user just said:
        "{user_input}"

        Please rewrite it as a clear, specific, and standalone query suitable for searching music.
        Do not introduce new preferences or details that were not explicitly mentioned.
        Only output the rewritten query. Do not include any explanations or extra text.
        """.strip()

        response = self.embedding_processor.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        rewritten_query = response.choices[0].message.content.strip()
        return rewritten_query


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
        self.reranker = CohereReranker(model_name="rerank-english-v3.0")
    
    # for Score-based (加權融合)
    def ranking_score_based(self, query: str, top_k: int = 200) -> List[Dict]:
        # Get query embeddings
        clap_vector = self.embedding_processor.get_text_embedding(query, use_clap=True)
        openai_vector = self.embedding_processor.get_text_embedding(query, use_clap=False)
        
        # Search in both tables
        audio_results = self.db.search_songs(clap_vector, self.db.tables["audio"], top_k)
        text_results = self.db.search_songs(openai_vector, self.db.tables["text"], top_k)

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
            distances = df.loc[mask, "_distance"] # 取出對應來源的 distance (from searching LanceDB)，_distance = 1 - cosine_similarity，_distance 越小越相似
            # 取得來源之最大、最小距離用於 normalization
            min_val = distances.min()
            max_val = distances.max()
            norm_col = f"{source_name}_normalized_distance"
            # range 0~1
            df.loc[mask, norm_col] = (distances - min_val) / (max_val - min_val)
            return df
        # Do normalization for both audio and text
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
                "source": lambda x: ",".join(sorted(set(x))),
                "combined_info": lambda x: next((val for val in x if pd.notna(val)), None)
            })
            .reset_index()
        )

        # Compute weighted score per song (加權階段)
        def compute_weighted_score(row):
            audio_sim = 1 - row["audio_normalized_distance"] if pd.notna(row["audio_normalized_distance"]) else None # 1 - distance, 再做一次反轉，越大越相似
            text_sim = 1 - row["text_normalized_distance"] if pd.notna(row["text_normalized_distance"]) else None # 1 - distance, 再做一次反轉，越大越相似
            w_audio = self.audio_weight if audio_sim is not None else 0
            w_text = self.text_weight if text_sim is not None else 0
            total = w_audio + w_text if (w_audio + w_text) > 0 else 1
            return ((audio_sim or 0) * w_audio + (text_sim or 0) * w_text) / total

        grouped["score"] = grouped.apply(compute_weighted_score, axis=1)

        # Extract individual raw distances
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

        # === Rerank using Cohere ===
        rerank_top_n = 10  # choose n to do rerank
        top_songs = grouped.sort_values("score", ascending=False).head(rerank_top_n)
        subset = text_df[text_df["song_name"].isin(top_songs["song_name"])]
        
        subset = subset.copy()
        subset_records = []
        for _, row in subset.iterrows():            
            record = {
                "song_name": row["song_name"],
                "text": row["combined_info"],
                "text_vector": row["text_vector"]
            }
            subset_records.append(record)

        subset_schema = pa.schema([
            pa.field("song_name", pa.string()),
            pa.field("text", pa.string()),
            pa.field("text_vector", pa.list_(pa.float32(), 1536))
        ])

        temp_db = lancedb.connect("./.lancedb_temp")
        if "rerank_tmp" in temp_db.table_names():
            temp_db.drop_table("rerank_tmp")
        temp_table = temp_db.create_table("rerank_tmp", schema = subset_schema, mode="overwrite")
        temp_table.add(subset_records)
        results = temp_table.search(openai_vector, vector_column_name = 'text_vector').rerank(reranker=self.reranker, query_string=query).to_df()
        
        # Merge rerank results with original data
        top_songs = top_songs.merge(results[["song_name", "_relevance_score"]], on="song_name", how="left")
        top_songs = top_songs.sort_values("_relevance_score", ascending=False)
        
        # Format final output for LLM explanation or downstream use
        final_recommendations = []
        final_recommendations_rerank = []
        # for _, row in top_songs.sort_values("_relevance_score", ascending=False).head(rerank_top_n).iterrows():
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
                "audio_distance": row["audio_distance"],
                "text_distance": row["text_distance"],
                "similarity_score": row["score"],
                "similarity_audio": 1 - row["audio_normalized_distance"],
                "similarity_text": 1 - row["text_normalized_distance"],
                "source": row["source"],
                #"rerank_score": row["_relevance_score"],
                "audio_path": row["audio_path"],
                "combined_info": row["combined_info"]
            })
            
        for _, row in top_songs.sort_values("_relevance_score", ascending=False).head(rerank_top_n).iterrows():
            final_recommendations_rerank.append({
                "song_name": row["song_name"],
                "artist": row["artist"],
                "mood": row["mood"],
                "video_theme": row["video_theme"],
                "genre": row["genre"],
                "instrument": row["instrument"],
                "other_tags": row["other_tags"],
                "description": row["lmm_description"],
                # This is from LanceDB return
                "similarity_score": row["score"],
                "similarity_audio": 1 - row["audio_normalized_distance"],
                "similarity_text": 1 - row["text_normalized_distance"],
                "source": row["source"],
                "rerank_score": row["_relevance_score"],
                "audio_path": row["audio_path"],
                "combined_info": row["combined_info"]
            })

        return final_recommendations, final_recommendations_rerank
    
    # usefull function because its result is not good
    def ranking_text_then_audio_rerank(self, query: str, top_k: int = 20) -> List[Dict]:
        # text then audio rerank
        clap_vector = self.embedding_processor.get_text_embedding(query, use_clap=True)
        openai_vector = self.embedding_processor.get_text_embedding(query, use_clap=False)

        # Search in text tables for 20 results
        text_results = self.db.search_songs(openai_vector, self.db.tables["text"], top_k)
        text_results = text_results.copy()
        text_results["source"] = "text"

        # candidate is the song name searched in text table
        candidate_names = text_results["song_name"].tolist()
        audio_results = self.db.search_songs(clap_vector, self.db.tables["audio"], top_k=200)
        audio_results = audio_results[audio_results["song_name"].isin(candidate_names)].copy()
        audio_results["audio_score"] = 1 - audio_results["_distance"]

        search_db = lancedb.connect("./.lancedb_2")
        text_df = search_db.open_table("music_text").to_pandas()
        metadata_columns = ["artist", "lmm_description"]
        merged = audio_results.merge(text_df[["song_name"] + metadata_columns], on="song_name", how="left")
        # merged = merged.merge(metadata_table, on="song_name", how="left")

        merged["score"] = merged["audio_score"]

        final_recommendations = []
        for _, row in merged.sort_values("score", ascending=False).iterrows():
            # audio_path = self.db.generate_audio_path(row["artist"], row["song_name"])
            final_recommendations.append({
                "song_name": row["song_name"],
                "artist": row["artist"],
                "description": row["lmm_description"],
                "similarity_score": row["score"],
                #"similarity_audio": row.get("audio_score", 0),
                #"similarity_text": row.get("text_score", 0),
                #"sorce": "text,audio" if pd.notna(row.get("audio_score")) else "text",
                #"audio_path": audio_path if os.path.exists(audio_path) else None
            })

        return final_recommendations

class RankingEvaluator:
    def __init__(self, k: int):
        self.k = k

    def precision_at_k(self, gt, recs):
        recs_at_k = recs[:self.k]
        relevant = [r for r in recs_at_k if r in gt]
        return len(relevant) / self.k

    def recall_at_k(self, gt, recs):
        recs_at_k = recs[:self.k]
        relevant = [r for r in recs_at_k if r in gt]
        return len(relevant) / len(gt) if len(gt) > 0 else 0.0

    def ndcg_at_k(self, gt, recs):
        recs_at_k = recs[:self.k]
        dcg = 0.0
        for i, rec in enumerate(recs_at_k):
            if rec in gt:
                dcg += 1 / np.log2(i + 2)  # log2(i+2) since i starts from 0
        ideal_rels = [1] * min(len(gt), self.k)
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels)])
        return dcg / idcg if idcg > 0 else 0.00

    def average_precision(self, gt, recs):
        hits = 0
        precisions = []
        for i, rec in enumerate(recs):
            if rec in gt:
                hits += 1
                precisions.append(hits / (i + 1))
        return sum(precisions) / len(gt) if gt else 0.0

    def evaluate(self, gt, recs, method_name):
        return {
            'Method': method_name,
            f'Precision@{self.k}': self.precision_at_k(gt, recs),
            f'Recall@{self.k}': self.recall_at_k(gt, recs),
            f'nDCG@{self.k}': self.ndcg_at_k(gt, recs),
            'MAP': self.average_precision(gt, recs)
        }

    def evaluate_random_baseline(self, gt, pool, n):
        precision_scores, recall_scores, ndcg_scores, map_scores = [], [], [], []

        for _ in range(n):
            recs = random.sample(pool, self.k)
            precision_scores.append(self.precision_at_k(gt, recs))
            recall_scores.append(self.recall_at_k(gt, recs))
            ndcg_scores.append(self.ndcg_at_k(gt, recs))
            map_scores.append(self.average_precision(gt, recs))

        return {
            'Method': 'Baseline (Random)',
            f'Precision@{self.k}': np.mean(precision_scores),
            f'Recall@{self.k}': np.mean(recall_scores),
            f'nDCG@{self.k}': np.mean(ndcg_scores),
            'MAP': np.mean(map_scores)
        }
