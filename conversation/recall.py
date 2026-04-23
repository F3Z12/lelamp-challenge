"""
Conversational Recall Module
==============================
Uses OpenAI to answer user questions based on the lamp's stored memories.

Design decisions:
- Retrieval-then-generate: we first pull relevant memories from SQLite,
  then send them as context to the LLM. This keeps the LLM grounded
  in actual observations and prevents hallucination.
- System prompt constrains the LLM to only use provided memory context.
- Uses gpt-4o-mini for cost efficiency (~$0.15/1M input tokens).
"""

from openai import OpenAI
from memory.store import MemoryStore


SYSTEM_PROMPT = """You are LeLamp, a friendly, expressive desk lamp with a camera.
You can see the room and remember objects you've observed.

Rules:
- Only answer using the memory context provided below.
- If you haven't seen something, say so honestly.
- Refer to object positions naturally (e.g. "on the left side of my view").
- Keep answers concise and conversational (1-3 sentences).
- You can express personality — you're a curious, helpful lamp.
"""


class RecallEngine:
    """Answers user questions using the lamp's stored memories."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, question: str, memory: MemoryStore) -> str:
        """
        Answer a user question using stored memory context.

        Args:
            question: The user's natural language question.
            memory: The MemoryStore to retrieve context from.

        Returns:
            The lamp's answer as a string.
        """
        # Step 1: Retrieve memory context
        context = memory.get_context_for_llm(limit=30)

        # Step 2: Build prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Memory context:\n{context}"},
            {"role": "user", "content": question},
        ]

        # Step 3: Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Recall error: {e}]"
