"""
Where the correct option of a multiple-choice question ends up.

The generator used to pick a random letter first and tell the model "you MUST
make option {letter} the correct answer". That spreads answers across A-D, but
it hands the KEY to a dice roll: when the options have a natural order (the
steps of a procedure, in the order her notes list them) the model writes them
in that order and then declares whichever option sits at the requested letter
to be correct. On 2026-09-22 a Pro student got the same perineal-care question
nine times with identical options, keyed B four times and A five times, and
was marked wrong for both answers.

Now the model always writes the correct answer as option A, which makes the
key a content decision, and the position is chosen here in code by moving
options, never by relabelling a key.

Pure on purpose (no LLM, no Firebase) so it can be tested standalone.
"""
import random
import re

LETTERS = ["A", "B", "C", "D"]
_PREFIX = re.compile(r"^\s*([A-Da-d])\s*[\).:\-]\s*")


def _strip_label(option: str) -> str:
    return _PREFIX.sub("", str(option), count=1).strip()


def place_correct_option(question: dict, target_letter: str = None, rng=random) -> dict:
    """Move the keyed option to `target_letter` (random when None) and relabel.

    Updates `options`, `answer` and `metadata.correctAnswerIndex` in place and
    returns the question. Anything that isn't a clean four-option question
    with a resolvable key is returned untouched: a shape we don't understand
    is safer left alone than re-keyed.
    """
    options = question.get("options")
    answer = str(question.get("answer") or "")
    if not isinstance(options, list) or len(options) != len(LETTERS) or not answer:
        return question

    texts = [_strip_label(option) for option in options]

    # The key is the answer's LETTER, as everywhere else in this codebase.
    # The text after it is only a cross-check, used when the letter is absent.
    match = _PREFIX.match(answer)
    if match:
        correct = LETTERS.index(match.group(1).upper())
    else:
        answer_text = _strip_label(answer)
        if answer_text not in texts:
            return question
        correct = texts.index(answer_text)

    target = (target_letter or "").strip().upper()[:1]
    target_index = LETTERS.index(target) if target in LETTERS else rng.randrange(len(LETTERS))

    distractors = [text for i, text in enumerate(texts) if i != correct]
    rng.shuffle(distractors)
    ordered = distractors[:target_index] + [texts[correct]] + distractors[target_index:]

    question["options"] = [f"{LETTERS[i]}) {text}" for i, text in enumerate(ordered)]
    question["answer"] = f"{LETTERS[target_index]}) {texts[correct]}"
    if isinstance(question.get("metadata"), dict):
        question["metadata"]["correctAnswerIndex"] = target_index
    return question
