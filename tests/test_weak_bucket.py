# -*- coding: utf-8 -*-
"""
Weak-bucket selection: which skill the post-node insight names, if any.

These exist because of a bug that hid the endpoint's best finding. acc()
returns None for an unsampled bucket and a float otherwise, and the guards
used to read `(acc(...) or 1) < 0.55` to mean "an unsampled bucket cannot be
the weak one". 0.0 is falsy in Python too, so a bucket the student got
ENTIRELY wrong evaluated to 1.0 and was dropped — the student who missed every
single select-all, the clearest weakness this endpoint can observe, was the one
student it had nothing to say to.

The first test below is that case. The rest pin the surrounding conditions so
the fix cannot be traded back for a shorter line.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_weak_bucket.py

main.py builds LLM clients at import time and needs API keys, so we load the
selection block out of the source file rather than importing the module.
"""
import io
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── load just the selection block, without importing main.py ────────────────
SRC = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()

start = SRC.index("    def acc(d):")
end = SRC.index("    if not (strong_bucket and weak_bucket):")
BLOCK = textwrap.dedent(SRC[start:end])

# It lives inside `async def node_debrief`, so it reads `combined` from the
# enclosing scope and assigns the two names we want back out.
CODE = compile(BLOCK, "main.py:buckets", "exec")


def select(knowledge, priority, multi):
    """Run the real block over one node's tallies. -> (strong, weak)"""
    def bucket(pair):
        correct, total = pair
        return {"correct": correct, "total": total}

    ns = {"combined": {
        "knowledge": bucket(knowledge),
        "priority": bucket(priority),
        "multi": bucket(multi),
    }}
    exec(CODE, ns)  # noqa: S102
    return ns["strong_bucket"], ns["weak_bucket"]


failures = []


def check(name, got, want):
    if got != want:
        failures.append("%s\n    expected: %r\n    got:      %r" % (name, want, got))


# ── the regression this file exists for ─────────────────────────────────────
# She knows the content and misses every select-all. Before the fix this
# returned (knowledge, None) and the screen said it was still figuring her out.
check(
    "a bucket she got entirely wrong IS the weak bucket",
    select(knowledge=(3, 3), priority=(0, 0), multi=(0, 3)),
    ("knowledge", "multi"),
)

check(
    "the same holds for prioritization",
    select(knowledge=(4, 4), priority=(0, 3), multi=(0, 0)),
    ("knowledge", "priority"),
)

# The neighbouring case that always worked — kept so a future rewrite that
# breaks one and not the other is caught.
check(
    "one right out of three is still weak",
    select(knowledge=(3, 3), priority=(0, 0), multi=(1, 3)),
    ("knowledge", "multi"),
)

# ── the strong side ─────────────────────────────────────────────────────────
check(
    "no strong side means no pattern, however bad the weak side is",
    # Insight requires BOTH: without demonstrated strength this is just
    # "you're bad at this", which is not a finding.
    select(knowledge=(1, 3), priority=(0, 0), multi=(0, 3)),
    (None, "multi"),
)

check(
    "a perfect knowledge score still counts as strong",
    select(knowledge=(5, 5), priority=(0, 0), multi=(1, 4)),
    ("knowledge", "multi"),
)

check(
    "75% is the strong threshold and it is inclusive",
    select(knowledge=(3, 4), priority=(0, 0), multi=(0, 3)),
    ("knowledge", "multi"),
)

check(
    "just under 75% is not strong",
    select(knowledge=(2, 3), priority=(0, 0), multi=(0, 3)),
    (None, "multi"),
)

# ── sampling floors ─────────────────────────────────────────────────────────
check(
    "an unsampled bucket is never the weak one",
    # This is what the `or 1` was reaching for, and it still holds.
    select(knowledge=(3, 3), priority=(0, 0), multi=(0, 0)),
    ("knowledge", None),
)

check(
    "under-sampled weakness does not qualify",
    # 0 of 2 is a bad afternoon, not a pattern. MIN_WEAK is 3.
    select(knowledge=(3, 3), priority=(0, 0), multi=(0, 2)),
    ("knowledge", None),
)

check(
    "under-sampled strength does not qualify",
    select(knowledge=(2, 2), priority=(0, 0), multi=(0, 3)),
    (None, "multi"),
)

check(
    "55% or above is not weak enough to name",
    select(knowledge=(4, 4), priority=(0, 0), multi=(2, 3)),
    ("knowledge", None),
)

# ── which weakness gets named when both qualify ─────────────────────────────
check(
    "the weaker of two candidates wins",
    select(knowledge=(4, 4), priority=(0, 4), multi=(1, 4)),
    ("knowledge", "priority"),
)

check(
    "and it is genuinely ordered, not just first-listed",
    select(knowledge=(4, 4), priority=(1, 4), multi=(0, 4)),
    ("knowledge", "multi"),
)


# ── report ──────────────────────────────────────────────────────────────────
print("")
if failures:
    print("FAILED (%d)" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)

print("weak bucket: all checks passed")
