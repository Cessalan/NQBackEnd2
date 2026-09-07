"""
Read-only Firestore scan: active users in a recent window.

Usage (from NQBackEnd2 root):
    venv\\Scripts\\python tools\\active_recent.py --days 21
    venv\\Scripts\\python tools\\active_recent.py --days 21 --weekly

WHY NOT tools/active_users.py

That script defines an active day as one with at least one message where
role == "user". For this product that is the wrong definition and it
undercounts badly: four of the ten paying subscribers have never typed a
single message. They drive the product entirely through buttons — upload,
generate, answer, continue — and every one of those actions writes an
ASSISTANT message, not a user one.

So here "active" means any message in a chat you own, either role. An
assistant message only exists because the student did something to cause it.

Both numbers are printed side by side so the difference is visible rather
than assumed.

Never writes to Firestore. Reads metadata fields only (role, timestamp,
userId, tier) — no message bodies.
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAGE_SIZE = 5000


def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(ROOT, "FireBaseAccess.json"))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def paginate(query):
    query = query.order_by("__name__").limit(PAGE_SIZE)
    last = None
    while True:
        q = query.start_after(last) if last else query
        docs = list(q.stream())
        if not docs:
            return
        yield from docs
        last = docs[-1]
        if len(docs) < PAGE_SIZE:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--weekly", action="store_true", help="break the window into 7-day buckets")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)
    db = get_db()

    # ── chat -> owner ────────────────────────────────────────────────────
    print("Scanning chats ...", flush=True)
    chat_owner = {}
    for doc in paginate(db.collection("chats").select(["userId"])):
        d = doc.to_dict() or {}
        chat_owner[doc.id] = d.get("userId") or "_unknown"
    print(f"  {len(chat_owner)} chats", flush=True)

    # ── tier per user ────────────────────────────────────────────────────
    tier = {}
    created = {}
    for doc in paginate(db.collection("users").select(["usage", "createdAt"])):
        d = doc.to_dict() or {}
        tier[doc.id] = ((d.get("usage") or {}).get("tier")) or "free"
        if d.get("createdAt"):
            created[doc.id] = d["createdAt"]
    print(f"  {len(tier)} user docs", flush=True)

    # ── messages ─────────────────────────────────────────────────────────
    # Filtered in Python rather than with a where() so this needs no
    # collection-group index on `timestamp`.
    print("Scanning messages (the big one) ...", flush=True)
    any_days = defaultdict(set)     # active days, either role
    typed_days = defaultdict(set)   # active days, user-authored only
    first_seen = {}
    scanned = 0
    for doc in paginate(db.collection_group("messages").select(["role", "timestamp", "hidden"])):
        scanned += 1
        if scanned % 25000 == 0:
            print(f"  {scanned} messages ...", flush=True)
        d = doc.to_dict() or {}
        ts = d.get("timestamp")
        if ts is None or d.get("hidden"):
            continue
        uid = chat_owner.get(doc.reference.parent.parent.id)
        if uid is None:
            continue

        tsu = ts.astimezone(timezone.utc)
        if uid not in first_seen or tsu < first_seen[uid]:
            first_seen[uid] = tsu
        if tsu < since:
            continue

        day = tsu.date().isoformat()
        any_days[uid].add(day)
        if d.get("role") == "user":
            typed_days[uid].add(day)
    print(f"  {scanned} messages scanned\n", flush=True)

    active = set(any_days)
    typed = set(typed_days)
    pro = {u for u in active if tier.get(u) == "pro"}
    returning = {u for u in active if len(any_days[u]) > 1}
    # New = first ever activity falls inside the window.
    new_users = {u for u in active if first_seen.get(u, now) >= since}

    print("=" * 62)
    print(f"ACTIVE USERS — last {args.days} days (since {since.date().isoformat()})")
    print("=" * 62)
    print(f"  Active (any activity)        : {len(active)}")
    print(f"  Active (typed a message)     : {len(typed)}"
          f"   <- what active_users.py counts")
    print(f"  Button-only (never typed)    : {len(active - typed)}")
    print()
    print(f"  New in window                : {len(new_users)}")
    print(f"  Returning (2+ distinct days) : {len(returning)}"
          f"  ({(100.0*len(returning)/len(active)):.1f}% of active)" if active else "")
    print(f"  Pro among active             : {len(pro)}")
    print(f"  Total users ever (user docs) : {len(tier)}")

    if active:
        buckets = defaultdict(int)
        for u in active:
            buckets[len(any_days[u])] += 1
        print("\n  Active days per user:")
        for n in sorted(buckets):
            label = f"{n} day" + ("s" if n > 1 else "")
            print(f"    {label:<10}{buckets[n]:>5} users")

    if args.weekly:
        print("\n" + "-" * 62)
        print("BY WEEK (most recent first)")
        print("-" * 62)
        for w in range(0, args.days, 7):
            hi = now - timedelta(days=w)
            lo = now - timedelta(days=min(w + 7, args.days))
            wk = {u for u, days in any_days.items()
                  if any(lo.date().isoformat() <= d <= hi.date().isoformat() for d in days)}
            wkpro = {u for u in wk if tier.get(u) == "pro"}
            print(f"  {lo.date()} → {hi.date()} : {len(wk):>4} active, {len(wkpro)} pro")


if __name__ == "__main__":
    main()
