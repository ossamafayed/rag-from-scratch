"""Wiring retriever + LLM together.

This is where RAG actually happens. The pattern is always the same:

    question -> retriever finds relevant chunks
             -> chunks are pasted into a prompt as {context}
             -> LLM answers using only that context

Built with LCEL (LangChain Expression Language): the `|` operator chains
steps together, the same way a shell pipe does.
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from rag.llm import get_llm
from rag.store import get_retriever

# Telling the model to say "I don't know" is what separates a grounded RAG
# answer from a confident hallucination.
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Use only the information in the context below to answer the question.
If the context does not contain the answer, say you don't know.
Do not invent details.

Context:
{context}

Question: {question}

Answer:"""
)

# Turns a follow-up like "what does it cover?" into a standalone question,
# so the retriever has real words to search with instead of "it".
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation and a follow-up question, rewrite the follow-up
as a standalone question that makes sense on its own. Output only the question.

Conversation:
{chat_history}

Follow-up: {question}
Standalone question:"""
)


def format_docs(docs: list[Document]) -> str:
    """Flatten retrieved chunks into the single string the prompt expects."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(k: int | None = None):
    """A one-shot question-answering chain. No memory of previous turns.

    RunnableParallel runs both branches on the same input at once:
      - "context"  : question -> retriever -> formatted string
      - "question" : the question, passed through untouched
    Its output dict feeds the prompt's two placeholders.
    """
    retriever = get_retriever(k)
    return (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | ANSWER_PROMPT
        | get_llm()
        | StrOutputParser()   # pull the plain string out of the model response
    )


def build_qa_chain_with_sources(k: int | None = None):
    """Same as build_qa_chain, but also returns the chunks used.

    Being able to show your sources is most of the point of RAG - it is how a
    user checks that the answer is real.
    """
    retriever = get_retriever(k)
    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["documents"]))
        | ANSWER_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return RunnableParallel(
        documents=retriever,
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=answer)


def format_history(history: list[tuple[str, str]]) -> str:
    """Render [(question, answer), ...] as plain text for the condense prompt."""
    return "\n".join(f"Human: {q}\nAssistant: {a}" for q, a in history)


class ChatRAG:
    """Conversational RAG: remembers the exchange so follow-ups work.

    Written as a small class instead of a LangChain memory object because the
    logic is only a few lines, and seeing it explicitly is the lesson:
    memory in RAG is just "rewrite the question, then do normal RAG".
    """

    def __init__(self, k: int | None = None):
        self.history: list[tuple[str, str]] = []
        self.qa = build_qa_chain(k)
        self.condenser = CONDENSE_PROMPT | get_llm() | StrOutputParser()

    def ask(self, question: str) -> str:
        # On the first turn there is nothing to condense against.
        if self.history:
            search_question = self.condenser.invoke(
                {
                    "chat_history": format_history(self.history),
                    "question": question,
                }
            ).strip()
        else:
            search_question = question

        answer = self.qa.invoke(search_question)
        self.history.append((question, answer))
        return answer

    def reset(self) -> None:
        self.history.clear()
