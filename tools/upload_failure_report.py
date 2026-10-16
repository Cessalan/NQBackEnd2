"""
Read-only Firestore scan: why uploads did not complete.

Usage (from NQBackEnd2 root):
    venv\\Scripts\\python tools\\upload_failure_report.py
    venv\\Scripts\\python tools\\upload_failure_report.py --days 7 --ids

WHY THIS EXISTS

On 2026-10-13 about 60 of ~70 failed uploads in three weeks had left nothing
in the Cloud Run error log, and funnelEvents only had upload_started and
upload_completed, so a failure was a missing row with no reason. The frontend
now writes `upload_failed` (the page saw the error; `reason` says which) and
`upload_interrupted` (the page died mid-upload: `via` pagehide or next_load).

COUNTING RULES

- One upload is one funnelId. An interrupted upload can write two rows
  (pagehide, then next_load), so everything counts distinct funnelIds.
- An upload that completed is never counted as failed, even if a stray
  interruption row exists for it.
- `--ids` prints funnelIds; grep Cloud Run logs for `upload:<id>` to see the
  server's side of the same upload.

Never writes to Firestore.
"""

import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import firebase_admin
from firebase_admin import credentials, firestore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OWNER_UID = "VymBNcl0KiOrv1tbPPQgjHal0SG3"
STEPS = ["upload_started", "upload_completed", "upload_failed", "upload_interrupted"]


def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(ROOT, "FireBaseAccess.json"))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def fetch(db, days):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (db.collection("funnelEvents")
              .where(filter=firestore.FieldFilter("createdAt", ">=", since))
              .stream())
    return [r for r in (d.to_dict() for d in rows)
            if r.get("step") in STEPS and r.get("uid") != OWNER_UID and r.get("funnelId")]


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


def bucket_seconds(ms):
    if ms is None:
        return "?"
    s = ms / 1000
    for limit in (10, 30, 60, 120, 300):
        if s < limit:
            return f"<{limit}s"
    return "300s+"


def report(rows, show_ids):
    by_funnel = defaultdict(list)
    for r in rows:
        by_funnel[r["funnelId"]].append(r)

    started = {f for f, rs in by_funnel.items() if any(r["step"] == "upload_started" for r in rs)}
    completed = {f for f, rs in by_funnel.items() if any(r["step"] == "upload_completed" for r in rs)}
    failed = {f: next(r for r in rs if r["step"] == "upload_failed")
              for f, rs in by_funnel.items() if f not in completed and any(r["step"] == "upload_failed" for r in rs)}
    interrupted = {f: [r for r in rs if r["step"] == "upload_interrupted"]
                   for f, rs in by_funnel.items()
                   if f not in completed and f not in failed and any(r["step"] == "upload_interrupted" for r in rs)}
    silent = started - completed - set(failed) - set(interrupted)

    print(f"Uploads started:      {len(started)}")
    print(f"  completed:          {len(completed & started)}  ({pct(len(completed & started), len(started))})")
    print(f"  failed, reason seen: {len(failed)}")
    print(f"  interrupted:        {len(interrupted)}")
    print(f"  no outcome at all:  {len(silent)}  (old clients, or the student never came back)")

    print("\nFailed, by reason")
    reasons = Counter(r.get("reason") for r in failed.values())
    for reason, n in reasons.most_common():
        sample = [r for r in failed.values() if r.get("reason") == reason]
        hidden = sum(1 for r in sample if r.get("wasHidden"))
        multi = sum(1 for r in sample if (r.get("fileCount") or 0) >= 4)
        print(f"  {n:4}  {reason:24} hidden-tab {hidden}, 4+ files {multi}")

    print("\nFailed, by phase / last server event")
    for (phase, last), n in Counter((r.get("phase"), r.get("lastEvent")) for r in failed.values()).most_common(10):
        print(f"  {n:4}  {phase} / {last}")

    print("\nFailed, by elapsed time")
    for b, n in sorted(Counter(bucket_seconds(r.get("elapsedMs")) for r in failed.values()).items()):
        print(f"  {n:4}  {b}")

    print("\nInterrupted, by last server event and how it was reported")
    for (last, vias), n in Counter(
            (next((r.get("lastEvent") for r in rs if r.get("lastEvent")), None),
             "+".join(sorted({r.get("via") or "?" for r in rs}))) for rs in interrupted.values()).most_common(10):
        print(f"  {n:4}  {last}  via {vias}")

    if show_ids:
        print("\nfunnelIds (grep Cloud Run logs for upload:<id>)")
        for f, r in failed.items():
            print(f"  failed       {f}  {r.get('reason')}")
        for f in interrupted:
            print(f"  interrupted  {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--ids", action="store_true")
    args = parser.parse_args()
    report(fetch(get_db(), args.days), args.ids)


if __name__ == "__main__":
    main()
