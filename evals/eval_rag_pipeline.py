import json
import os
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from deepeval.models import GeminiModel
from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
#JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

judge_model = GeminiModel(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
    
)
JUDGE_MODEL = judge_model

# 1. LOAD queries (we only need the queries — context comes from the pipeline now)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["query"])          # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],       # what the generator produced
            retrieval_context=result["context"],  # what the RETRIEVER returned
        )
    )


# 3. THE THREE TRIAD METRICS
metrics = [
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


# 4. EVALUATE
evaluate(test_cases=test_cases, metrics=metrics)

# in this we evalutate with RAG TRIAD = ContextualRelevancy, Faithfulness, AnswerRelevancy