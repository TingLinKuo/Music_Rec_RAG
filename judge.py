from google import genai
from typing_extensions import TypedDict
from google.genai import types
import json
from enum import Enum
from typing_extensions import TypedDict
from collections import Counter
import glob

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval
)
from deepeval.test_case import LLMTestCaseParams
from pydantic import BaseModel
from anthropic import Anthropic
from deepeval.models import DeepEvalBaseLLM
import instructor

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

# model for using
GEMINI_2_FLASH = "gemini-2.0-flash"
GEMINI_1_5_PRO = "gemini-1.5-pro"
GEMINI_2_5_PRO = "gemini-2.5-pro-preview-05-06" #Pre-release version

def run_vote(client, model_version, pair_content, sys_prompt, n):
    votes = []
    comments = []
    
    for _ in range(n):
        response = client.models.generate_content(
            model=model_version,
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

def run_score(client, model_version, content, sys_prompt, n):    
    scores = []
    comments = []
    
    for _ in range(n):
        response = client.models.generate_content(
            model=model_version,
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


# generation evaluation

class CustomClaudeSonnet(DeepEvalBaseLLM):
    def __init__(self):
        self.model = Anthropic()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_anthropic(client)
        resp = instructor_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Claude-3.5-Sonnet"


def evaluate_all_metrics_with_usefulness(
    query: str,
    generated_output: str,
    context_docs: list[str],
    thresholds: dict = None,
    model: str = "o4-mini"
) -> dict:
    """
    Evaluate LLM-generated explanation using four metrics:
    - Faithfulness
    - Answer Relevancy
    - Usefulness (custom GEval)

    Returns:
        dict: Each metric's score, pass/fail status, and reason.
    """
    thresholds = thresholds or {
        "faithfulness": 0.7,
        "relevancy": 0.7,
        "usefulness": 0.7
    }

    test_case = LLMTestCase(
        input=query,
        actual_output=generated_output,
        retrieval_context=context_docs,
    )

    # === Metric definitions ===
    faithfulness_metric = FaithfulnessMetric(
        threshold=thresholds["faithfulness"],
        model=model,
        include_reason=True
    )

    relevancy_metric = AnswerRelevancyMetric(
        threshold=thresholds["relevancy"],
        model=model,
        include_reason=True
    )

    usefulness_metric = GEval(
        name="Usefulness",
        criteria="Evaluate whether the explanation helps the user understand why these songs were recommended.",
        evaluation_steps=[
            "The input is the user's query. The actual output is the explanation generated by the model for a music recommendation.",
            "Determine whether the explanation helps the user understand how the recommended music relates to their request, or clearly states if no strong connection exists.",
            "If the user's query is vague, abstract, or unclear, the explanation should acknowledge that ambiguity. Such responses are acceptable and can be rated positively.",
            "If the explanation describes music that does not clearly match the user's request, it should point out the mismatch or uncertainty honestly. These cases can still be considered helpful if handled transparently.",
            "Penalize explanations that fabricate or exaggerate connections, or that rely on vague, repetitive, or uninformative language.",
            "Reward explanations that are specific, logically structured, honest, and informative, helping the user understand the rationale behind the recommendation.",
            "Do not penalize based on tone, formatting, or writing style. Only evaluate the explanation's content and its usefulness."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        threshold=thresholds["usefulness"],
        model=model
    )

    # === Run evaluation ===
    results_raw = evaluate(
        test_cases=[test_case],
        metrics=[faithfulness_metric, relevancy_metric, usefulness_metric]
    )

    return results_raw
