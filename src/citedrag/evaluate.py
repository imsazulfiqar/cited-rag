"""Run the golden question set end to end and write a report.

Retrieval metrics are computed for every question at no cost. With --judge,
each question is also answered by Claude and scored by the LLM judge.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .metrics import citation_precision, hit_at_k, reciprocal_rank, safe_mean
from .retrieve import Retriever


def load_golden(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_eval(retriever: Retriever, golden: list[dict], k: int = 5, use_llm: bool = False,
             client=None) -> dict:
    if use_llm:
        from .generate import has_api_key
        if client is None and not has_api_key():
            raise SystemExit("--judge needs ANTHROPIC_API_KEY (retrieval metrics work without it)")
        from .generate import answer as generate_answer
        from .judge import judge

    rows = []
    for item in golden:
        expected = set(item["expected_docs"])
        hits = retriever.search(item["question"], k=k)
        docs = [h.chunk.doc_id for h in hits]
        row = {
            "id": item["id"],
            "question": item["question"],
            "retrieved": docs,
            "hit@1": hit_at_k(docs, expected, 1),
            f"hit@{k}": hit_at_k(docs, expected, k),
            "mrr": reciprocal_rank(docs, expected),
        }
        if use_llm:
            ans = generate_answer(item["question"], hits, client=client)
            cited = [c.chunk_id.split(":")[0] for c in ans.citations]
            context = "\n\n".join(f"[{h.chunk.label}] {h.chunk.text}" for h in hits)
            scores = judge(item["question"], item["reference_answer"], ans.text, context, client=client)
            row.update({
                "answer": ans.text,
                "citation_precision": citation_precision(cited, expected),
                "refused": ans.refused,
                **({"relevance": scores.relevance, "accuracy": scores.accuracy,
                    "groundedness": scores.groundedness, "rationale": scores.rationale} if scores else {}),
            })
        rows.append(row)

    summary_keys = ["hit@1", f"hit@{k}", "mrr"]
    if use_llm:
        summary_keys += ["citation_precision", "relevance", "accuracy", "groundedness"]
    summary = {key: safe_mean(r.get(key) for r in rows) for key in summary_keys}
    return {"summary": summary, "rows": rows, "k": k, "n": len(rows)}


def write_report(result: dict, out_dir: str | Path = "eval/results") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [f"# Evaluation report ({stamp} UTC)", "", f"Questions: {result['n']}", "",
             "| Metric | Score |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in result["summary"].items()]
    lines += ["", "## Per question", "", "| id | hit@1 | mrr | retrieved |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['id']} | {r['hit@1']} | {r['mrr']:.2f} | {', '.join(r['retrieved'][:3])} |"
              for r in result["rows"]]
    path = out / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
