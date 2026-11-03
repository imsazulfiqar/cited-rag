"""LLM-as-judge: score a generated answer against a reference answer.

Scores are returned as validated structured output, so the evaluation loop
never has to regex numbers out of free text.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from .config import FALLBACK_BETA, settings

JUDGE_PROMPT = """You are grading an answer produced by a document question-answering system.

Question:
{question}

Reference answer (written by a human expert):
{reference}

Source excerpts the system was given:
{context}

System answer:
{answer}

Score each dimension from 1 (poor) to 5 (excellent):
- relevance: does it answer the question that was asked, for the right market/scope?
- accuracy: do its facts agree with the reference answer?
- groundedness: is every claim supported by the source excerpts (no invented facts)?
Give a one-sentence rationale."""


class JudgeScores(BaseModel):
    relevance: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    rationale: str


def judge(question: str, reference: str, answer: str, context: str,
          client: anthropic.Anthropic | None = None) -> JudgeScores | None:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.parse(
        model=settings.judge_model,
        max_tokens=16000,
        output_config={"effort": "low"},
        betas=[FALLBACK_BETA],
        fallbacks="default",
        output_format=JudgeScores,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, context=context, answer=answer)}],
    )
    if response.stop_reason == "refusal":
        return None
    return response.parsed_output
