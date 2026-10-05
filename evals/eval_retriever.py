"""Evaluate vector retrieval after CrossEncoder reranking.

Run from the repository root with: python -m evals.eval_retriever
"""

import json
import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.models import GeminiModel

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

#from src.reranker import RerankingRetriever
from src.retriever import build_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
#JUDGE_MODEL = "gemini-2.5-flash"  
THRESHOLD = 0.7

judge_model = GeminiModel(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
    
)
JUDGE_MODEL = judge_model
# 1. LOAD the golden set --- the fixed, human-authored truth
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.
retriever = build_retriever()#RerankingRetriever(fetch_k=10, top_k=5)
test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )


# 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


# 4. EVALUATE --- every metric on every case, batched + parallel, with a printed report
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "cross_encoder_reranked",
        "embedding_model": "gemini-embedding-001",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "fetch_k": 10,
        "top_k": 5,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)