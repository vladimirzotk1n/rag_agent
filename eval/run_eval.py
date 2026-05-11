import asyncio
import json
import math
import random
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.agent.rephraser import rephrase_query
from src.agent.tools import make_retrieve_tool
from src.db.get_top import get_top
from src.utils.prompts import SYSTEM_PROMPT

load_dotenv()

N_EVAL = 30  # количество запросов на конфигурацию
TOP_K = 15
SEED = 42
DATA_PATH = "data/json/queried_tk_rf_flat.json"
RESULTS_DIR = Path("eval/results")

AgentMode = Literal["single", "multi"]
RetrievalMode = Literal["dense", "bm25", "hybrid"]

CONFIGS: list[tuple[AgentMode, RetrievalMode]] = [
    ("single", "dense"),
    ("single", "bm25"),
    ("single", "hybrid"),
    ("multi", "dense"),
    ("multi", "bm25"),
    ("multi", "hybrid"),
]


def load_samples(n: int) -> list[dict]:
    with open(DATA_PATH) as f:
        data = json.load(f)

    # подсчет количества чанков на статью
    article_counts: dict[int, int] = {}
    for d in data:
        article_counts[d["article"]] = article_counts.get(d["article"], 0) + 1

    test = [d for d in data if d.get("split") == "test"]
    random.seed(SEED)
    chosen = random.sample(test, min(n, len(test)))

    samples = []
    for chunk in chosen:
        qs = chunk.get("natural_queries", [])
        if not qs:
            continue
        q = random.choice(qs).removeprefix("query: ").strip()
        samples.append(
            {
                "question": q,
                "article_id": chunk["article"],
                "ground_truth": chunk["article_content"],
                "total_relevant": article_counts[chunk["article"]],
            }
        )
    return samples[:n]


def calc_retrieval_metrics(
    retrieved_ids: list[int], relevant: int, total_relevant: int
) -> dict:
    k = len(retrieved_ids)
    hits = [1 if i == relevant else 0 for i in retrieved_ids]
    hit = float(any(hits))
    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / total_relevant if total_relevant else 0.0
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    ndcg = dcg / ideal if ideal else 0.0
    return {"Hit@k": hit, "Precision@k": precision, "Recall@k": recall, "nDCG@k": ndcg}


async def run_single(
    question: str, mode: RetrievalMode, llm: ChatOpenAI
) -> tuple[str, list[dict]]:
    rephrased = await rephrase_query(question)
    hits = await get_top(query=question, sparse_query=rephrased, top_k=TOP_K, mode=mode)
    context = "\n\n".join(h["article_content"] for h in hits)
    prompt = (
        SYSTEM_PROMPT.substitute(context=context)
        + f"\n\nВопрос пользователя:\n{question}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    return resp.content, hits


async def run_multi(
    question: str, mode: RetrievalMode, llm: ChatOpenAI
) -> tuple[str, list[dict]]:
    collected: list[dict] = []
    agent = create_react_agent(model=llm, tools=[make_retrieve_tool(mode, collected)])
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content, collected


async def eval_config(
    agent_mode: AgentMode,
    retrieval_mode: RetrievalMode,
    samples: list[dict],
    llm: ChatOpenAI,
) -> dict:
    runner = run_single if agent_mode == "single" else run_multi

    ret_rows = []
    ragas_samples = []

    for i, s in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {s['question'][:70]}")
        try:
            answer, hits = await runner(s["question"], retrieval_mode, llm)
        except Exception as e:
            print(f"    SKIP — {e}")
            continue

        ids = [h.get("article") for h in hits]
        contexts = [h["article_content"] for h in hits]

        ret_rows.append(
            calc_retrieval_metrics(ids, s["article_id"], s["total_relevant"])
        )
        ragas_samples.append(
            SingleTurnSample(
                user_input=s["question"],
                response=answer,
                retrieved_contexts=contexts or [""],
                reference=s["ground_truth"],
            )
        )

    if not ret_rows:
        return {"agent_mode": agent_mode, "retrieval_mode": retrieval_mode}

    ret_avg = {k: sum(r[k] for r in ret_rows) / len(ret_rows) for k in ret_rows[0]}

    dataset = EvaluationDataset(samples=ragas_samples)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    ragas_avg = ragas_result.to_pandas().mean(numeric_only=True).to_dict()

    return {
        "agent_mode": agent_mode,
        "retrieval_mode": retrieval_mode,
        **{f"retr_{k}": v for k, v in ret_avg.items()},
        **ragas_avg,
    }


async def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    samples = load_samples(N_EVAL)
    print(f"Loaded {len(samples)} eval samples  (TOP_K={TOP_K})")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    eval_emb = LangchainEmbeddingsWrapper(LCOpenAIEmbeddings())
    faithfulness.llm = eval_llm
    answer_relevancy.llm = eval_llm
    answer_relevancy.embeddings = eval_emb
    context_precision.llm = eval_llm
    context_recall.llm = eval_llm

    rows = []
    for agent_mode, retrieval_mode in CONFIGS:
        print(f"\n── {agent_mode} / {retrieval_mode} {'─' * 40}")
        row = await eval_config(agent_mode, retrieval_mode, samples, llm)
        rows.append(row)
        print(f" -> {row}")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "eval_results_15.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
