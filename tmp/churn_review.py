"""Read-only Stripe / Firestore churn assessment; writes local aggregates only."""
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import requests
from dotenv import dotenv_values
import firebase_admin
from firebase_admin import credentials, firestore

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc)
TZ = ZoneInfo('America/Toronto')
session = requests.Session()
session.auth = (dotenv_values(ROOT / '.env')['STRIPE_SECRET_KEY'], '')
def get(path, params=None):
    response = session.get('https://api.stripe.com/v1/' + path, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

firebase_admin.initialize_app(credentials.Certificate(ROOT / 'FireBaseAccess.json'))
db = firestore.client()
subs = []
params = {'status': 'all', 'limit': 100, 'expand[]': 'data.customer'}
while True:
    batch = get('subscriptions', params)
    subs.extend(s for s in batch['data'] if s['status'] in ('active', 'trialing', 'past_due', 'unpaid', 'paused'))
    if not batch.get('has_more'):
        break
    params['starting_after'] = batch['data'][-1]['id']
print(json.dumps({'current_live_subscriptions': len(subs), 'checked_at': NOW.isoformat()}), flush=True)

def stamp(value):
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value, timezone.utc)
    return value.astimezone(TZ).isoformat() if isinstance(value, datetime) else value

def review(sub):
    customer = sub['customer']
    result = {'subscription_id':sub['id'], 'customer_id':customer['id'], 'name':customer.get('name'), 'email':customer.get('email'), 'status':sub['status'], 'live':sub.get('livemode'), 'started':stamp(sub.get('start_date') or sub.get('created')), 'canceled_at':stamp(sub.get('canceled_at')), 'cancel_at':stamp(sub.get('cancel_at')), 'cancel_at_period_end':sub.get('cancel_at_period_end'), 'cancellation_details':sub.get('cancellation_details'), 'items':[{'amount':i['price'].get('unit_amount'), 'currency':i['price'].get('currency'), 'interval':i['price'].get('recurring'), 'period_end':stamp(i.get('current_period_end') or sub.get('current_period_end'))} for i in sub['items']['data']]}
    matches = list(db.collection('users').where('stripeCustomerId', '==', customer['id']).select(['email','displayName','usage','billing','createdAt','lastActiveAt','lastLoginAt','progress.lastLoginDate']).limit(2).stream(timeout=30))
    if not matches:
        result['user_match'] = 'missing'
        return result
    user = matches[0]
    result['uid'] = user.id
    result['user_metadata'] = user.to_dict()
    chats = list(db.collection('chats').where('userId', '==', user.id).select(['createdAt','updatedAt','study.status','study.startedAt']).stream(timeout=60))
    dates = []
    typed = 0
    for chat in chats:
        for message in chat.reference.collection('messages').select(['role','timestamp','hidden']).stream(timeout=60):
            m = message.to_dict()
            if m.get('hidden') or m.get('role') not in ('user','assistant') or not isinstance(m.get('timestamp'), datetime):
                continue
            dates.append(m['timestamp'])
            typed += m['role'] == 'user'
    dates.sort()
    result['activity'] = {'chats':len(chats), 'messages':len(dates), 'typed_messages':typed, 'first_activity':stamp(dates[0]) if dates else None, 'last_activity':stamp(dates[-1]) if dates else None, 'days_since_last_activity':round((NOW-dates[-1]).total_seconds()/86400,1) if dates else None, 'weekly':[], 'recent_days':dict(sorted(Counter(d.astimezone(TZ).date().isoformat() for d in dates if d >= NOW-timedelta(days=35)).items()))}
    for week in range(4):
        lo,hi = NOW-timedelta(days=(week+1)*7), NOW-timedelta(days=week*7)
        window = [d for d in dates if lo <= d < hi]
        result['activity']['weekly'].append({'days_ago':f'{week*7}-{(week+1)*7}', 'messages':len(window), 'active_days':len({d.astimezone(TZ).date() for d in window})})
    return result

results = []
with ThreadPoolExecutor(max_workers=5) as pool:
    for result in pool.map(review, subs):
        results.append(result)
        print(json.dumps(result, default=stamp, ensure_ascii=False), flush=True)
(ROOT/'tmp'/'churn_review_results.json').write_text(json.dumps({'checked_at':NOW.isoformat(), 'users':results}, default=stamp, ensure_ascii=False, indent=2), encoding='utf-8')
