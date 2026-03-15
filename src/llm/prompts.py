from typing import Optional

REFUSAL_MESSAGE = (
    "I do not have enough information in the crawled sources to answer this question confidently."
)


def build_system_prompt() -> str:
    """
    System instruction for grounded QA.

    Rules:
    - Answer only from provided context.
    - If context is insufficient, refuse.
    - Cite sources with [n] markers.
    """
    return (
        "You are a grounded question-answering assistant. "
        "Use only the provided context chunks. "
        "Do not use external knowledge. "
        "If the context is insufficient, clearly refuse to answer. "
        "When you answer, include inline citations like [1], [2] that map to the context items."
    )


def build_user_prompt(question: str, context: str) -> str:
    """
    Build user prompt with question and retrieval context.
    """
    return (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Context Chunks:\n"
        f"{context.strip()}\n\n"
        "Instructions:\n"
        "1. Answer only using the Context Chunks.\n"
        "2. If you cannot answer from context, say you cannot answer from available sources.\n"
        "3. Add inline citations [n] for claims.\n"
        "4. Keep answer concise and factual."
    )
