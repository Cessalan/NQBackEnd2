"""
Read-only Firestore scan: who saw a paywall, when, and in what context.

Usage (from NQBackEnd2 root):
    venv\\Scripts\\python tools\\paywall_report.py
    venv\\Scripts\\python tools\\paywall_report.py --days 7
    venv\\Scripts\\python tools\\paywall_report.py --users     # per-person rows

WHY THIS EXISTS

The product could see that a couple of dozen people had ever reached Stripe,
but not how many were ever ASKED. Those two numbers imply opposite fixes:

    many views, few clicks   -> the copy or the timing is wrong
    few views                -> the gates simply never fire, and tightening
                                copy is wasted effort

Nothing in the app emitted a paywall event until `funnelEvents` shipped, so
that question was unanswerable and every decision about pricing and gating was
inference. This turns it into a query.

WHAT IT WILL NOT DO

It does not merge a WALL with a BROWSE. A student blocked by a quota and a
student who tapped the usage badge to look at the offer are different
populations; averaging them produces a conversion rate that describes nobody.
`trigger` and `blocked` keep them apart, and the report always splits on them.

Never writes to Firestore.
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

COLLECTION = "funnelEvents"

VIEW = "paywall_viewed"
DISMISS = "paywall_dismissed"
CLICK = "paywall_cta_clicked"
CHECKOUT = "checkout_started"


def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(ROOT, "FireBaseAccess.json"))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def fetch(db, days):
    """All funnel rows newer than `days`, oldest first."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    q = (db.collection(COLLECTION)
           .where("createdAt", ">=", since)
           .order_by("createdAt"))
    return [d.to_dict() for d in q.stream()]


def pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "—"


def report(rows, show_users):
    views = [r for r in rows if r.get("step") == VIEW]
    if not views:
        print("No paywall views recorded in this window.")
        print("That is itself the finding: the gates are not firing.")
        return

    clicks = [r for r in rows if r.get("step") in (CLICK, CHECKOUT)]
    dismissals = [r for r in rows if r.get("step") == DISMISS]

    # A person can be asked more than once; both numbers matter.
    people = {r.get("uid") for r in views if r.get("uid")}
    clickers = {r.get("uid") for r in clicks if r.get("uid")}

    print("=" * 66)
    print(f"PAYWALL VIEWS — {len(views)} views by {len(people)} people")
    print("=" * 66)
    print(f"  reached CTA / checkout : {len(clickers)} people  ({pct(len(clickers), len(people))})")
    print(f"  dismissed              : {len(dismissals)} times")

    # ── Wall vs browse ────────────────────────────────────────────────────
    # The single most important split in this file. A "browse" is someone
    # volunteering to look at the offer with budget still left.
    walls = [v for v in views if v.get("blocked")]
    browses = [v for v in views if not v.get("blocked")]
    print(f"\n  BLOCKED (a wall)       : {len(walls)} views")
    print(f"  NOT blocked (a browse) : {len(browses)} views")

    # ── What opened it ────────────────────────────────────────────────────
    by_trigger = defaultdict(lambda: {"views": 0, "people": set(), "clicked": 0})
    for v in views:
        t = by_trigger[v.get("trigger") or "unknown"]
        t["views"] += 1
        if v.get("uid"):
            t["people"].add(v["uid"])
            if v["uid"] in clickers:
                t["clicked"] += 1

    print("\n" + "-" * 66)
    print(f"{'TRIGGER':<26}{'VIEWS':>7}{'PEOPLE':>8}{'→CTA':>7}")
    print("-" * 66)
    for name, d in sorted(by_trigger.items(), key=lambda kv: -kv[1]["views"]):
        print(f"{name:<26}{d['views']:>7}{len(d['people']):>8}{d['clicked']:>7}")

    # ── How far she had got ───────────────────────────────────────────────
    by_step = defaultdict(int)
    for v in views:
        by_step[v.get("reachedStep") or "—"] += 1
    print("\n" + "-" * 66)
    print("HOW FAR SHE HAD GOT WHEN ASKED")
    print("-" * 66)
    for name, n in sorted(by_step.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<40}{n:>6}")

    # ── Urgency available? ────────────────────────────────────────────────
    # examDaysAway is null when no exam date was on the profile, which means
    # the urgency copy could not have rendered even though it exists.
    with_exam = [v for v in views if v.get("examDaysAway") is not None]
    soon = [v for v in with_exam if v["examDaysAway"] <= 7]
    print("\n" + "-" * 66)
    print("EXAM URGENCY")
    print("-" * 66)
    print(f"  views with a known exam date : {len(with_exam)}  ({pct(len(with_exam), len(views))})")
    print(f"  of those, exam within 7 days : {len(soon)}")
    if len(with_exam) == 0:
        print("  ⚠️  No view had an exam date — the urgency copy never rendered.")

    if show_users:
        print("\n" + "=" * 66)
        print("PER-VIEW DETAIL")
        print("=" * 66)
        for v in views:
            ts = v.get("createdAt")
            when = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else "?"
            uid = (v.get("uid") or "anon")[:12]
            mark = "CLICKED" if v.get("uid") in clickers else ""
            print(f"{when}  {uid:<13}{v.get('trigger', '?'):<20}"
                  f"{'WALL' if v.get('blocked') else 'browse':<8}"
                  f"q={v.get('questionsRemaining', '?')}/{v.get('questionsLimit', '?')} "
                  f"p={v.get('plansRemaining', '?')}/{v.get('planLimit', '?')} "
                  f"exam={v.get('examDaysAway', '—')} {mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="lookback window (default 30)")
    ap.add_argument("--users", action="store_true", help="print one line per view")
    args = ap.parse_args()

    rows = fetch(get_db(), args.days)
    print(f"Scanned {len(rows)} funnel rows from the last {args.days} days.\n")
    report(rows, args.users)


if __name__ == "__main__":
    main()
