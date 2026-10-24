"""A RAG chatbot with guardrails, built by fixing three separate leaks.

`ChatRAG` in chain.py is the textbook version. It hallucinates on real
conversations, and it is worth being precise about WHERE:

  LEAK 1 - the condenser invents questions.
      Told to "rewrite the follow-up as a standalone question", it is given
      "ok thanks" and produces "What is the purpose of the Smoking Policy?".
      The user never asked that. Everything downstream then answers it
      correctly and confidently - the damage was done before retrieval.

  LEAK 2 - retrieval always returns something.
      There is no "no results". An off-topic question still gets the 4
      nearest chunks, and they get pasted into the prompt as if relevant.

  LEAK 3 - chit-chat goes down the document pipeline.
      "hi" is not a question about company policy, but nothing in the
      pipeline is able to notice that.

Each guard below closes one leak. They are independent - you can enable and
disable them one at a time, which is what step 10 does.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import config
from rag.chain import ANSWER_PROMPT, format_docs, format_history
from rag.llm import get_llm
from rag.router import CHITCHAT, chitchat_reply, classify
from rag.store import load_store

# Beyond this cosine distance we treat the match as "not in the document".
# Calibrated from step 4's real numbers on this corpus:
#   on-topic question      ~0.3 - 0.6
#   related but absent     ~1.0 - 1.5
#   completely off-topic   ~1.8+
# There is no universal value - it depends on the embedding model and your
# documents. Measure yours with step 4 rather than copying this number.
DISTANCE_THRESHOLD = 1.2

# GUARD 1: the condenser gets an explicit escape hatch. The original prompt
# offered no way to say "this isn't a follow-up question", so it always
# produced a question. Now doing nothing is a valid, named option.
SAFE_CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """You rewrite follow-up questions so they can stand alone.

Rules:
- If the message is a genuine question that depends on the conversation
  (it uses "it", "that", "they", or is otherwise incomplete), rewrite it as a
  standalone question.
- If the message is already a complete question, repeat it unchanged.
- If the message is NOT a question at all - a greeting, thanks, an
  acknowledgement, small talk - output exactly: NOT_A_QUESTION
- Never invent a question the user did not ask.

Conversation:
{chat_history}

Message: {question}
Output:"""
)

NOT_A_QUESTION = "NOT_A_QUESTION"

REFUSAL = (
    "I don't have anything about that in the document. "
    "Try asking about a company policy - conduct, recruitment, internet and "
    "email, mobile phones, smoking, drugs and alcohol, health and safety, "
    "discrimination and harassment, or discipline and termination."
)


class GuardedChatRAG:
    """Conversational RAG with routing, safe condensing and a relevance gate.

    Every guard can be switched off so you can see what each one prevents.
    """

    def __init__(
        self,
        k: int | None = None,
        *,
        route: bool = True,
        safe_condense: bool = True,
        gate: bool = True,
        threshold: float = DISTANCE_THRESHOLD,
    ):
        self.history: list[tuple[str, str]] = []
        self.k = k or config.TOP_K
        self.route = route
        self.safe_condense = safe_condense
        self.gate = gate
        self.threshold = threshold

        self.store = load_store()
        self.llm = get_llm()
        self.condenser = SAFE_CONDENSE_PROMPT | self.llm | StrOutputParser()
        self.answerer = ANSWER_PROMPT | self.llm | StrOutputParser()

        # Filled in on each ask() so callers (and step 10) can show the work.
        self.trace: dict = {}

    def ask(self, message: str) -> str:
        self.trace = {"message": message}

        # --- GUARD 3: route before doing anything else ---------------------
        routed_as_document = False
        if self.route:
            intent = classify(message)
            self.trace["intent"] = intent
            routed_as_document = intent != CHITCHAT
            if intent == CHITCHAT:
                answer = chitchat_reply(message)
                self.trace["path"] = "chitchat (no retrieval, no LLM)"
                # Deliberately NOT added to history: small talk is not context
                # for a later follow-up, and keeping it only gives the
                # condenser more material to misread.
                self.trace["answer"] = answer
                return answer

        # --- GUARD 1: condense only when it is genuinely a follow-up -------
        search_query = message
        if self.history:
            if self.safe_condense:
                rewritten = self.condenser.invoke(
                    {"chat_history": format_history(self.history), "question": message}
                ).strip()
                if NOT_A_QUESTION in rewritten.upper():
                    # Two guards, one disagreement. The router already looked
                    # at this message in isolation and called it a question -
                    # a judgement made without the conversation history to
                    # confuse it - so it wins. Falling back to the message
                    # unchanged is always safe: worst case we search with a
                    # slightly under-specified query, which GUARD 2 can still
                    # catch. Refusing a real question is the costlier error.
                    if routed_as_document:
                        self.trace["condenser_overruled"] = True
                        search_query = message
                    else:
                        self.trace["path"] = "condenser said NOT_A_QUESTION"
                        answer = chitchat_reply(message)
                        self.trace["answer"] = answer
                        return answer
                else:
                    search_query = rewritten
            else:
                from rag.chain import CONDENSE_PROMPT

                search_query = (CONDENSE_PROMPT | self.llm | StrOutputParser()).invoke(
                    {"chat_history": format_history(self.history), "question": message}
                ).strip()
        self.trace["searched_as"] = search_query

        # --- GUARD 2: check the match is actually close enough --------------
        hits = self.store.similarity_search_with_score(search_query, k=self.k)
        best = hits[0][1] if hits else float("inf")
        self.trace["best_distance"] = best

        if self.gate and best > self.threshold:
            self.trace["path"] = f"gated (distance {best:.3f} > {self.threshold})"
            self.trace["answer"] = REFUSAL
            return REFUSAL

        # --- answer, grounded in the retrieved chunks -----------------------
        docs = [doc for doc, _score in hits]
        answer = self.answerer.invoke(
            {"context": format_docs(docs), "question": search_query}
        ).strip()

        self.trace["path"] = "answered from document"
        self.trace["answer"] = answer
        self.history.append((message, answer))
        return answer

    def reset(self) -> None:
        self.history.clear()
