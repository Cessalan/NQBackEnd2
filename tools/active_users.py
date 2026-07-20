"""
Read-only Firestore scan: rank users by how often they come back.

Usage (from NQBackEnd2 root):
    venv\Scripts\python tools\active_users.py            # use cache if present
    venv\Scripts\python tools\active_users.py --refresh  # re-scan Firestore

- Active day  = UTC calendar day with at least one user-authored message.
- Return count = distinct active days - 1 (primary ranking metric).
- Never writes to Firestore. No message bodies are read or stored:
  the scan selects only role/timestamp/hidden fields.
- Aggregates cached to tools/active_users_cache.json (gitignored).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import timezone

import firebase_admin
from firebase_admin import credentials, firestore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_PATH = os.path.join(HERE, "active_users_cache.json")
PAGE_SIZE = 5000


def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(ROOT, "FireBaseAccess.json"))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def paginate(query):
    """Yield docs from a query in pages, using __name__ cursors."""
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


def sync():
    db = get_db()

    # Map chatId -> (userId, userEmail). Metadata fields only.
    print("Scanning chats ...", flush=True)
    chat_owner = {}
    chat_email = {}
    chats_per_user = defaultdict(int)
    for doc in paginate(db.collection("chats").select(["userId", "userEmail"])):
        d = doc.to_dict() or {}
        uid = d.get("userId") or "_unknown"
        chat_owner[doc.id] = uid
        chats_per_user[uid] += 1
        if d.get("userEmail"):
            chat_email[uid] = d["userEmail"]
    print(f"  {len(chat_owner)} chats, {len(chats_per_user)} distinct users", flush=True)

    # Optional: emails from the users collection (small, metadata only).
    for doc in paginate(db.collection("users").select(["email"])):
        d = doc.to_dict() or {}
        if d.get("email") and doc.id not in chat_email:
            chat_email[doc.id] = d["email"]

    # Scan every message once, keeping only role/timestamp/hidden.
    print("Scanning messages (this is the big one) ...", flush=True)
    active_days = defaultdict(set)
    msg_count = defaultdict(int)
    scanned = 0
    for doc in paginate(db.collection_group("messages").select(["role", "timestamp", "hidden"])):
        scanned += 1
        if scanned % 20000 == 0:
            print(f"  {scanned} messages ...", flush=True)
        d = doc.to_dict() or {}
        if d.get("role") != "user" or d.get("hidden"):
            continue
        chat_id = doc.reference.parent.parent.id
        uid = chat_owner.get(chat_id)
        if uid is None:
            continue
        ts = d.get("timestamp")
        if ts is None:
            continue
        day = ts.astimezone(timezone.utc).date().isoformat()
        active_days[uid].add(day)
        msg_count[uid] += 1
    print(f"  {scanned} messages scanned", flush=True)

    users = []
    for uid, days in active_days.items():
        ordered = sorted(days)
        users.append({
            "userId": uid,
            "email": chat_email.get(uid, ""),
            "activeDays": len(ordered),
            "returnCount": len(ordered) - 1,
            "firstSeen": ordered[0],
            "lastSeen": ordered[-1],
            "userMessages": msg_count[uid],
            "chats": chats_per_user.get(uid, 0),
        })
    users.sort(key=lambda u: (-u["returnCount"], -u["userMessages"]))

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"users": users, "messagesScanned": scanned}, f, indent=2)
    print(f"Cached aggregates to {CACHE_PATH}")
    return users


def report(users, top=30):
    total = len(users)
    returners = sum(1 for u in users if u["returnCount"] >= 1)
    print()
    print(f"{total} users with at least one message; "
          f"{returners} ({returners * 100 // max(total, 1)}%) came back on 2+ days.")
    print()
    hdr = f"{'#':>3}  {'user':<38} {'days':>4} {'rtrn':>4} {'msgs':>5} {'chats':>5}  {'first seen':<10}  {'last seen':<10}"
    print(hdr)
    print("-" * len(hdr))
    for i, u in enumerate(users[:top], 1):
        who = u["email"] or u["userId"]
        print(f"{i:>3}  {who[:38]:<38} {u['activeDays']:>4} {u['returnCount']:>4} "
              f"{u['userMessages']:>5} {u['chats']:>5}  {u['firstSeen']}  {u['lastSeen']}")
    if total > top:
        print(f"... {total - top} more in {CACHE_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-scan Firestore instead of using the cache")
    ap.add_argument("--top", type=int, default=30, help="how many rows to print")
    args = ap.parse_args()

    if not args.refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            users = json.load(f)["users"]
        print(f"(from cache {CACHE_PATH}; pass --refresh to re-scan)")
    else:
        users = sync()
    report(users, args.top)


if __name__ == "__main__":
    sys.exit(main())
