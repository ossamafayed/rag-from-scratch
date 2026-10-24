"""STEP 10 - Why the chatbot hallucinates on "ok thanks", and how to stop it.

Run:  python steps/10_stop_hallucinating.py

Type small talk into step 8 and the bot answers a policy question you never
asked. This step reproduces that, shows exactly which component invented the
question, then switches the guards on one at a time.

The lesson: the hallucination does NOT come from the answering model. The
answer prompt is grounded and behaves. It comes from the CONDENSER, upstream,
which was told to always produce a question and had no way to decline.
"""

import _bootstrap  # noqa: F401

from rag.chain import ChatRAG, format_history
from rag.guarded import DISTANCE_THRESHOLD, GuardedChatRAG
from rag.llm import check_ollama
from rag.store import load_store

CONVERSATION = [
    "What is the smoking policy?",   # a real question, to seed the history
    "ok thanks",                     # not a question
    "hi how u doin",                 # not a question
    "no more",                       # not a question
]


def part_1_reproduce() -> None:
    print("#" * 72)
    print("# PART 1 - the bug, and where it actually comes from")
    print("#" * 72)
    print()

    chat = ChatRAG()
    store = load_store()

    for message in CONVERSATION:
        # Recompute the rewrite so we can print it. This is the same call the
        # chain makes internally - we are just looking at it.
        if chat.history:
            searched = chat.condenser.invoke(
                {"chat_history": format_history(chat.history), "question": message}
            ).strip()
        else:
            searched = message

        distance = store.similarity_search_with_score(searched, k=1)[0][1]
        answer = chat.ask(message).strip()

        print(f"YOU TYPED : {message!r}")
        print(f"SEARCHED  : {searched!r}")
        print(f"DISTANCE  : {distance:.3f}")
        print(f"ANSWER    : {answer[:160]}")
        print("-" * 72)

    print()
    print("Look at the SEARCHED lines. The user never asked any of those")
    print("questions - the condenser manufactured them. It was instructed to")
    print("'rewrite the follow-up as a standalone question' and given no way")
    print("to say 'that wasn't a question', so it complied.")
    print()
    print("Note the distances too: some are LOW. A relevance threshold alone")
    print("would not have caught this, because the invented question really")
    print("is about the document. You have to stop it being invented.")
    print()


def part_2_guards() -> None:
    print("#" * 72)
    print("# PART 2 - adding the guards one at a time")
    print("#" * 72)
    print()

    configs = [
        ("no guards (same as step 8)",
         dict(route=False, safe_condense=False, gate=False)),
        ("+ GUARD 1: condenser may answer NOT_A_QUESTION",
         dict(route=False, safe_condense=True, gate=False)),
        ("+ GUARD 2: relevance gate on retrieval distance",
         dict(route=False, safe_condense=True, gate=True)),
        ("+ GUARD 3: route chit-chat away from RAG entirely",
         dict(route=True, safe_condense=True, gate=True)),
    ]

    for label, flags in configs:
        print("=" * 72)
        print(label)
        print("=" * 72)
        chat = GuardedChatRAG(**flags)
        for message in CONVERSATION:
            answer = chat.ask(message)
            trace = chat.trace
            detail = trace.get("searched_as", "-")
            print(f"  {message!r}")
            print(f"     path     : {trace.get('path')}")
            if "searched_as" in trace:
                print(f"     searched : {detail!r}  (distance "
                      f"{trace.get('best_distance', float('nan')):.3f})")
            print(f"     answer   : {answer.strip()[:120]}")
        print()


def part_3_still_works() -> None:
    print("#" * 72)
    print("# PART 3 - the guards must not break real questions")
    print("#" * 72)
    print()
    print("A guard that refuses everything is not a fix. Real follow-ups,")
    print("including ones that depend on history, still have to work.\n")

    chat = GuardedChatRAG()
    for message in [
        "What is the mobile phone policy?",
        "What does it cover?",              # genuine follow-up, needs history
        "thanks!",                          # small talk between real questions
        "And what happens if I lose the phone?",
    ]:
        answer = chat.ask(message)
        print(f"  {message!r}")
        print(f"     path   : {chat.trace.get('path')}")
        print(f"     answer : {answer.strip()[:200]}")
        print()


def main() -> None:
    check_ollama()
    part_1_reproduce()
    part_2_guards()
    part_3_still_works()

    print("#" * 72)
    print("# SUMMARY - the recipe")
    print("#" * 72)
    print("""
1. ROUTE FIRST. Decide what the message is before doing RAG on it.
   Greetings and thanks never touch the retriever. Cheapest and biggest win.

2. LET EVERY LLM STEP SAY NO. Any prompt that must always produce output will
   invent output. The condenser needed an explicit NOT_A_QUESTION escape
   hatch. This applies to every LLM call in a pipeline, not just the last one.

3. GATE ON DISTANCE. Retrieval always returns its nearest neighbours, however
   far. If the closest chunk is beyond your threshold (here {threshold}),
   refuse instead of calling the LLM. Measure the threshold on your own
   corpus with step 4 - do not copy a number from a tutorial.

4. KEEP SMALL TALK OUT OF HISTORY. Anything in history becomes material the
   condenser can misread on the next turn.

5. ANSWER FROM CONTEXT ONLY. Step 6's grounded prompt. This is the last line
   of defence, not the first - by the time a bad question reaches it, it is
   already answering the wrong thing correctly.

The general principle: in a multi-step LLM pipeline, an error at step 1 gets
faithfully executed by every step after it. Debug from the FRONT, and print
the intermediate values - that is how the condenser was caught here.
""".format(threshold=DISTANCE_THRESHOLD))


if __name__ == "__main__":
    main()
