from google import genai
from typing_extensions import TypedDict
from google.genai import types
import json
from enum import Enum
from typing_extensions import TypedDict
from collections import Counter
import glob

class Choice(Enum):
    first = "0"
    second = "1"
    third = "2"

class PairingOverall(TypedDict):
    overall_choice: Choice
    overall_comment: str

class ScoringOverall(TypedDict):
    overall_score: int
    overall_comment: str

def find_file_path(artist, song_name, base_dir="music"):
    pattern = f"{base_dir}/{artist} - {song_name}.*"
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def model_pair_content(modolity, query, audio1, audio2):
    input = f'Here is the query of {modolity} to music retrieval task:\nThis is a/an {modolity} . Please evaluate the following two background music based on this {modolity}. Which one is more suitable?'
    query = query
    response_prompt1 = "\nHere is the first reponse:\n"
    audio1 = audio1
    response_prompt2 = "\nHere is the second reponse:\n"
    audio2 = audio2
    content = []
    content.append(input)
    content.append(query)
    content.append(response_prompt1)
    content.append(audio1)
    content.append(response_prompt2)
    content.append(audio2)
  
    return content

def run_vote(client, pair_content, sys_prompt, n):
    votes = []
    comments = []
    
    for _ in range(n):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=pair_content,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=PairingOverall
            )
        )
        result = json.loads(response.text)
        votes.append(result['overall_choice'])
        comments.append(result['overall_comment'])

    return votes, comments

def majority_vote(votes):
    vote_count = Counter(votes)
    winner = vote_count.most_common(1)[0][0]  # '0', '1', or '2'
    return winner, dict(vote_count)

def score_content(modolity, query, audio):
    input = f'Here is the query of {modolity} to music retrieval task:\nThis is a/an {modolity} . Please evaluate the following background music based on this {modolity} and give a score.'
    query = query
    response_prompt = "\nHere is the reponse:\n"
    content = []
    content.append(input)
    content.append(query)
    content.append(response_prompt)
    content.append(audio)

    return content

def run_score(client, content, sys_prompt, n):    
    scores = []
    comments = []
    
    for _ in range(n):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=ScoringOverall
            )
        )
        result = json.loads(response.text)
        scores.append(result['overall_score'])
        comments.append(result['overall_comment'])

    return scores, comments

def majority_score(scores):
    score_average = sum(scores) / len(scores)
    score_count = Counter(scores)
    return score_average, dict(score_count)
