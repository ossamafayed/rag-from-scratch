"""STEP 6 - The prompt decides whether you get facts or fiction.

Run:  python steps/06_prompt_matters.py

Same retriever, same model, same question - three different prompts.
Only the instructions change.

Honest expectation: a modern instruction-tuned model like llama3.2 is already
fairly cautious, so the NAIVE prompt often behaves well anyway. That is luck,
not a guarantee - it depends on the model, the question, and the temperature.
The GROUNDED prompt is what makes it a property of your system. The RECKLESS
prompt shows what the model will happily do when you permit it.
"""

import _bootstrap  # noqa: F401

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from rag.chain import format_docs
from rag.llm import check_ollama, get_llm
from rag.store import get_retriever

# What you write when you aren't thinking about grounding at all.
NAIVE_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful HR assistant. Answer the employee's question.

{context}

Question: {question}
Answer:"""
)

# The same thing, but explicitly allowed to fall back on what it "knows".
# Nobody writes this on purpose - it is what a naive prompt can DRIFT into.
RECKLESS_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful HR assistant. Answer the employee's question.
Use the context if it helps, and otherwise answer from your general knowledge
of standard company policies. Always give a specific, confident answer.

{context}

Question: {question}
Answer:"""
)

# The guardrail.
GROUNDED_PROMPT = ChatPromptTemplate.from_template(
    """Use only the information in the context below to answer the question.
If the context does not contain the answer, reply exactly: "I don't know."
Do not use outside knowledge and do not guess.

Context:
{context}

Question: {question}
Answer:"""
)

# Not covered by the company policies document, so a grounded system must
# refuse. Anything else is invention.
QUESTION = "What is the company's parental leave allowance in weeks?"


def build(prompt):
    return (
        RunnableParallel(
            context=get_retriever() | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def main() -> None:
    check_ollama()
    print(f"QUESTION: {QUESTION}")
    print("(the document says nothing about parental leave)\n")

    prompts = [
        ("NAIVE prompt - no instruction about grounding", NAIVE_PROMPT),
        ("RECKLESS prompt - explicitly allows outside knowledge", RECKLESS_PROMPT),
        ("GROUNDED prompt - must answer from context or refuse", GROUNDED_PROMPT),
    ]
    for label, prompt in prompts:
        print("=" * 70)
        print(label)
        print("=" * 70)
        print(build(prompt).invoke(QUESTION).strip())
        print()

    print("-" * 70)
    print("Did any answer state a specific number of weeks? If so, that number")
    print("came from the model's training data, not your document - a")
    print("hallucination, delivered in exactly the same confident tone as a")
    print("real answer. That is what makes it dangerous.")
    print()
    print("Only the GROUNDED prompt makes refusal a guarantee rather than a")
    print("hope. In production you would also add a distance threshold: if the")
    print("nearest chunk is further away than some cutoff (step 4 showed the")
    print("off-topic query sitting at ~1.85), skip the LLM call entirely.")
    print("\nNext: python steps/07_chat_with_memory.py")


if __name__ == "__main__":
    main()
