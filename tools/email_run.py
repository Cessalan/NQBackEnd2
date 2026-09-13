"""
Run an email campaign from a terminal, or just look at who it would reach.

Usage (from NQBackEnd2 root):
    venv\\Scripts\\python tools\\email_run.py --preflight
    venv\\Scripts\\python tools\\email_run.py --audience
    venv\\Scripts\\python tools\\email_run.py --audience --campaigns   # slow
    venv\\Scripts\\python tools\\email_run.py --campaign plan_unstarted
    venv\\Scripts\\python tools\\email_run.py --campaign plan_unstarted --limit 5
    venv\\Scripts\\python tools\\email_run.py --campaign announcement --slug course_research

WHY THIS EXISTS

The only other trigger is `POST /api/email/run`, which needs the server running
and the cron secret in a header. That is the right shape for Cloud Scheduler and
the wrong shape for a person deciding whether to send anything at all. The
decision to mail two thousand people should not be taken through an endpoint
that is awkward to call, because awkward-to-call means rarely rehearsed, and the
first rehearsal should not be the real send.

So this runs exactly the same code paths as the route. Same selectors, same
guards, same idempotency, same log. Nothing here is a test harness.

THE CONFIRMATION

If EMAIL_ENABLED is not "true" this is a dry run and prints the payloads. If it
IS "true" then --yes is required, and without it the tool tells you precisely
what it was about to do and stops. That second gate exists because the flag is
usually set days before the send, in a file you are no longer looking at, so by
the time you run a campaign "am I live?" is a question you have stopped asking.

Reads .env the same way main.py does, so the flag you set there is the flag
this obeys.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

# Same order as main.py: .env, then .env.local wins.
load_dotenv(os.path.join(ROOT, ".env"))
load_dotenv(os.path.join(ROOT, ".env.local"), override=True)

import firebase_admin
from firebase_admin import credentials, firestore

from services import email_announcements as A
from services import email_campaigns as C
from services import email_sender as S


def _utf8_stdout():
    """Campaign output carries em dashes and accented topic names, and this
    console is cp1252. Without this the tool dies mid-run on a student whose
    subject happens to contain one."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.path.join(ROOT, "FireBaseAccess.json"))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def show_preflight(db):
    print("\nCONFIG")
    for k, v in S.preflight().items():
        print(f"  {k:<26} {v}")

    print("\n  sent today                 "
          f"{S.sent_today(db)} of {S.daily_cap()}")

    print("\nANNOUNCEMENTS")
    for a in A.listing():
        mark = "SENDABLE" if a["sendable"] else "blocked"
        print(f"  {a['slug']:<20} {a['status']:<9} audience={a['audience']:<9} {mark}")
        if not a["sendable"]:
            print(f"      {a['blocked_because']}")

    # The things that will not stop a send but should stop you.
    warn = []
    if not S.preflight()["postal_address_set"]:
        warn.append("EMAIL_POSTAL_ADDRESS is unset. CAN-SPAM requires a physical "
                    "address in every commercial message, and the footer will "
                    "print a red placeholder instead of one.")
    if not S.preflight()["token_secret_dedicated"]:
        warn.append("EMAIL_TOKEN_SECRET is unset, so unsubscribe links are signed "
                    "with the Stripe webhook secret. Rotating that secret would "
                    "silently invalidate every opt-out link already delivered.")
    if not S.preflight()["api_key_present"] and S.is_enabled():
        warn.append("EMAIL_ENABLED is true but RESEND_API_KEY is missing. Every "
                    "send will fail.")
    if warn:
        print("\nWARNINGS")
        for w in warn:
            print(f"  - {w}")


def show_audience(db, include_campaigns):
    rep = C.audience_report(db, include_campaigns=include_campaigns)
    print(f"\nusers scanned: {rep['users']}")
    print(f"sent today:    {rep['sent_today']} of {rep['daily_cap']}")

    print("\nAUDIENCES  (announcement targets)")
    for name, n in rep["audiences"].items():
        print(f"  {name:<10} {n}")

    if include_campaigns:
        print("\nCAMPAIGNS  (how many the selector would pick right now)")
        for name, n in rep["campaigns"].items():
            print(f"  {name:<16} {n}")
    else:
        print("\n  (pass --campaigns to also run each selector; it is minutes of reads)")


def run_campaign(db, campaign, slug, limit, yes):
    runner = C.CAMPAIGN_RUNNERS.get(campaign)
    if not runner:
        print(f"unknown campaign: {campaign}")
        print(f"known: {', '.join(sorted(C.CAMPAIGN_RUNNERS))}")
        return 2

    live = S.is_enabled()
    limit = limit if limit is not None else S.daily_cap()

    print(f"\ncampaign : {campaign}" + (f"  slug={slug}" if slug else ""))
    print(f"limit    : {limit}")
    print(f"mode     : {'LIVE — real messages' if live else 'dry run'}")
    print(f"budget   : {S.remaining_today(db)} left of {S.daily_cap()} today")

    if live and not yes:
        print("\nEMAIL_ENABLED is true, so this would send real email to real "
              "people and cannot be undone.")
        print("Re-run with --yes to go ahead, or unset EMAIL_ENABLED for a dry run.")
        return 1

    result = runner(db, limit=limit, slug=slug, verbose=True)

    print("\nRESULT")
    for k, v in result.items():
        if k == "detail":
            continue
        print(f"  {k:<14} {v}")

    if result.get("refused"):
        print("\n  Refused by the approval gate, which is it working. Flip the "
              "slug to approved in services/email_announcements.py when the "
              "copy is right.")
    return 0


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser(
        description="Run or inspect an email campaign. Dry run unless EMAIL_ENABLED=true.")
    ap.add_argument("--preflight", action="store_true",
                    help="config readiness and the announcement gate")
    ap.add_argument("--audience", action="store_true",
                    help="how many people each audience holds; sends nothing")
    ap.add_argument("--campaigns", action="store_true",
                    help="with --audience, also run each selector (slow)")
    ap.add_argument("--campaign", help=f"one of: {', '.join(sorted(C.CAMPAIGN_RUNNERS))}")
    ap.add_argument("--slug", help="which announcement, for --campaign announcement")
    ap.add_argument("--limit", type=int, help="max recipients (default: the daily cap)")
    ap.add_argument("--yes", action="store_true",
                    help="required to actually send when EMAIL_ENABLED=true")
    args = ap.parse_args()

    if not (args.preflight or args.audience or args.campaign):
        ap.print_help()
        return 0

    db = get_db()
    code = 0
    if args.preflight:
        show_preflight(db)
    if args.audience:
        show_audience(db, args.campaigns)
    if args.campaign:
        code = run_campaign(db, args.campaign, args.slug, args.limit, args.yes)
    return code


if __name__ == "__main__":
    sys.exit(main())
