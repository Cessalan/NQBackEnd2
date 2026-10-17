"""Did she get what she asked for? Read-only report over recent practice requests.

Two failures in one week (2026-10-11 and 2026-10-16) were found only because
someone opened the chat by hand: a Pro student asked for 15 and got 5, a free
student asked for a knowledge quiz and got nothing at all. This counts both
kinds from what the backend and frontend already save, so the next one shows
up here instead of in a refund request.

Sources:
  users/{uid}/practiceRequests/{id}  one per batch the server reserved quota
      for (services/practice_api.reserve): `count` promised, `questions`
      delivered, `status` pending until finish() runs.
  chats/{chatId}/messages            the quiz the frontend saved, if any.

Usage:
  python tools/delivery_report.py [--days 7] [--list 30] [--exclude-uid UID ...]

Pass the owner's uid to --exclude-uid: test chats otherwise read as failures.
"""
import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "service-account-key.json")
# A reservation still pending this long after it was made never reached finish().
STUCK_AFTER = timedelta(minutes=15)


def created(snap):
    """Server create time as an aware datetime (the SDK hands back a protobuf Timestamp)."""
    stamp = snap.create_time
    if stamp is None:
        return None
    return stamp.ToDatetime(tzinfo=timezone.utc) if hasattr(stamp, "ToDatetime") else stamp


def classify(data, created, now):
    promised = int(data.get("count") or 0)
    delivered = len(data.get("questions") or [])
    if data.get("status") != "complete":
        return "in_flight" if now - created < STUCK_AFTER else "stuck"
    if delivered == 0:
        return "empty"
    if delivered < promised:
        return "short"
    return "ok"


def saved_quiz(db, chat_id, since):
    """(saved quiz messages, orphan practice-only docs) in the chat since `since`."""
    saved, orphans = 0, 0
    for snap in db.collection("chats").document(chat_id).collection("messages").stream():
        at = created(snap)
        if at and at < since:
            continue
        msg = snap.to_dict() or {}
        if msg.get("type") == "quiz" and (msg.get("quizData") or msg.get("questions")):
            saved += 1
        elif msg.get("practice") and not msg.get("type"):
            orphans += 1
    return saved, orphans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--list", type=int, default=30, help="failed requests to list")
    parser.add_argument("--exclude-uid", action="append", default=[], help="skip this account (repeatable)")
    args = parser.parse_args()

    import firebase_admin
    from firebase_admin import credentials, firestore
    firebase_admin.initialize_app(credentials.Certificate(KEY))
    db = firestore.client()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)
    rows = []
    for snap in db.collection_group("practiceRequests").stream():
        at = created(snap)
        if not at or at < since:
            continue
        if snap.reference.parent.parent.id in args.exclude_uid:
            continue
        data = snap.to_dict() or {}
        rows.append({
            "uid": snap.reference.parent.parent.id,
            "chat": data.get("chat_id"),
            "at": at,
            "promised": int(data.get("count") or 0),
            "delivered": len(data.get("questions") or []),
            "pro": not data.get("charged") and int(data.get("count") or 0) > 0,
            "outcome": classify(data, at, now),
        })

    outcomes = Counter(r["outcome"] for r in rows)
    finished = [r for r in rows if r["outcome"] != "in_flight"]
    failed = [r for r in finished if r["outcome"] != "ok"]
    print(f"Practice requests, last {args.days} days: {len(rows)} "
          f"from {len({r['uid'] for r in rows})} students")
    for name in ("ok", "short", "empty", "stuck", "in_flight"):
        print(f"  {name:<10}{outcomes.get(name, 0)}")
    if finished:
        print(f"Got everything promised: {outcomes['ok'] / len(finished):.1%}")
        promised = sum(r["promised"] for r in finished)
        print(f"Questions promised {promised}, delivered {sum(r['delivered'] for r in finished)}")

    print(f"\nFailed requests (newest first, up to {args.list}); "
          "'saved' = quiz messages the chat has from that day on:")
    for r in sorted(failed, key=lambda r: r["at"], reverse=True)[:args.list]:
        saved, orphans = saved_quiz(db, r["chat"], r["at"] - timedelta(minutes=5)) if r["chat"] else (0, 0)
        print(f"  {r['at']:%m-%d %H:%M}  {r['outcome']:<6} {r['delivered']:>2}/{r['promised']:<3}"
              f" {'pro ' if r['pro'] else 'free'}  chat {r['chat']}  saved {saved}  orphan {orphans}")


if __name__ == "__main__":
    main()
