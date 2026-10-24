"""Deciding what a message even IS, before doing any RAG on it.

A naive RAG chatbot treats every incoming message as a document question.
"hi", "ok thanks" and "what is the smoking policy?" all get sent down the same
retrieve-then-answer pipeline, and the pipeline has no way to say "that wasn't
a question about the document".

The fix is a router: classify first, then act. This is the same idea as
routing in a web app - you don't send every HTTP request to the same handler.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.llm import get_llm

# Intents we handle.
CHITCHAT = "CHITCHAT"    # greetings, thanks, small talk
DOCUMENT = "DOCUMENT"    # a real question about the document

# Cheap deterministic shortcuts. Running these before the LLM makes the common
# cases instant and free, and - more importantly - makes them PREDICTABLE.
# A classifier you can't predict is a classifier you can't debug.
_GREETINGS = {
    "hi", "hey", "hello", "yo", "sup", "hiya", "howdy",
    "good morning", "good afternoon", "good evening",
    "how are you", "how r u", "how are u", "how u doin",
    "how are you doing", "hows it going", "how's it going",
}
_PLEASANTRIES = {
    "thanks", "thank you", "thx", "ty", "ok", "okay", "k", "cool", "nice",
    "great", "awesome", "got it", "understood", "sure", "yes", "no", "yep",
    "nope", "ok thanks", "okay thanks", "thanks!", "ok thank you",
    "no more", "nothing", "nothing else", "that's all", "thats all", "done",
    "bye", "goodbye", "see ya", "later",
}

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """Classify the user's message into exactly one label.

CHITCHAT - greetings, thanks, acknowledgements, farewells, small talk.
DOCUMENT - ANY request for information, including short follow-ups that
           depend on earlier context.

When in doubt, answer DOCUMENT. Wrongly treating a real question as small
talk is far worse than the reverse: the user gets no answer at all.

Examples:
Message: hello there
Label: CHITCHAT
Message: ok thanks, that helps
Label: CHITCHAT
Message: how are you doing today
Label: CHITCHAT
Message: what does it cover
Label: DOCUMENT
Message: And what happens if I lose the phone?
Label: DOCUMENT
Message: tell me more about that
Label: DOCUMENT

Reply with the single word CHITCHAT or DOCUMENT and nothing else.

Message: {message}
Label:"""
)


def normalise(message: str) -> str:
    """Lowercase and strip trailing punctuation, so 'Thanks!!' matches 'thanks'."""
    return message.lower().strip().strip("!?.,;: ")


def classify(message: str) -> str:
    """Return CHITCHAT or DOCUMENT for a single message.

    Deliberately does NOT look at conversation history: we are asking "what
    kind of message is this", not "what does it refer to". Mixing those two
    questions is exactly what made the naive condenser invent questions.
    """
    text = normalise(message)

    if not text:
        return CHITCHAT
    if text in _GREETINGS or text in _PLEASANTRIES:
        return CHITCHAT

    # A question mark means a question - unless it was one of the fixed
    # greetings above ("how are you?" is already handled). Checking this
    # BEFORE the LLM prevents the classifier's mistakes from silencing a
    # real question, which is the expensive direction to get wrong.
    if "?" in message:
        return DOCUMENT

    # Very short and no question mark: almost never a document question.
    if len(text.split()) <= 2:
        return CHITCHAT

    label = (CLASSIFY_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"message": message}
    )
    return CHITCHAT if CHITCHAT in label.upper() else DOCUMENT


def chitchat_reply(message: str) -> str:
    """A fixed, non-LLM response for small talk.

    Hard-coded on purpose. There is nothing to ground a generated reply
    against, so generating one is pure risk for no benefit.
    """
    text = normalise(message)
    if text in {"bye", "goodbye", "see ya", "later"}:
        return "Goodbye."
    # `startswith` so "hi how u doin" is recognised as a greeting, not just
    # the exact string "hi".
    if text in _GREETINGS or any(text.startswith(g) for g in _GREETINGS):
        return (
            "Hello. I answer questions about the company policies document - "
            "ask me anything in it."
        )
    return "You're welcome. Ask me anything else about the company policies."
