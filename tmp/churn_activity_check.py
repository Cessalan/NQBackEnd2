import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import firebase_admin
from firebase_admin import credentials, firestore
ROOT=Path(__file__).resolve().parents[1]
TZ=ZoneInfo('America/Toronto')
firebase_admin.initialize_app(credentials.Certificate(ROOT/'FireBaseAccess.json'))
db=firestore.client()
users=json.loads((ROOT/'tmp'/'churn_review_results.json').read_text(encoding='utf8'))['users']
def parse(v):
    if isinstance(v, datetime): return v.astimezone(timezone.utc)
    if isinstance(v,str):
        try: return datetime.fromisoformat(v.replace('Z','+00:00')).astimezone(timezone.utc)
        except ValueError: return None
    if isinstance(v,(int,float)):
        try: return datetime.fromtimestamp(v/1000 if v>1e11 else v, timezone.utc)
        except (ValueError,OSError): return None
    return None
def scan(u):
    ref=db.collection('users').document(u['uid'])
    activity=[]
    def add(v, source):
        d=parse(v)
        if d: activity.append((d,source))
    plans=[]
    for chat in db.collection('chats').where('userId','==',u['uid']).select(['title','updatedAt','study.status','study.startedAt','examDate','examName']).stream(timeout=60):
        c=chat.to_dict()
        add(c.get('updatedAt'),'chat.updatedAt')
        plans.append({'id':chat.id,**c})
        for m in chat.reference.collection('messages').select(['quizProgress.lastUpdated','flashcardProgress.lastUpdated','practice.updatedAt','practice.lastUpdated','createdAt']).stream(timeout=60):
            d=m.to_dict()
            for key in ('quizProgress','flashcardProgress','practice'):
                for field in ('lastUpdated','updatedAt'):
                    add((d.get(key) or {}).get(field),key+'.'+field)
    for p in ref.collection('studyPerformance').select(['updatedAt']).stream(timeout=60):
        add(p.to_dict().get('updatedAt'),'studyPerformance.updatedAt')
    latest=max(activity) if activity else None
    out={'name':u['name'],'latest_saved_progress':latest[0].astimezone(TZ).isoformat() if latest else None,'source':latest[1] if latest else None}
    if u['name'] in ('Jessica Martin','Kiewana Menefield','SANDY CASTILLO','Hailey  Hoye'):
        out['plans']=plans
        out['exams']=[e.to_dict() for e in ref.collection('exams').select(['name','date','examDate','examName','status']).stream(timeout=30)]
    return out
with ThreadPoolExecutor(max_workers=5) as pool:
    results=list(pool.map(scan,users))
for r in results: print(json.dumps(r,default=str,ensure_ascii=False),flush=True)
(ROOT/'tmp'/'churn_activity_check_results.json').write_text(json.dumps(results,default=str,ensure_ascii=False,indent=2),encoding='utf8')
