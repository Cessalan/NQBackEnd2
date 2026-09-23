"""Production admin API. All data and actions require a verified Firebase identity."""
import html
import os
from datetime import datetime, timezone
from uuid import uuid4
from services.email_html import sanitize_email_html
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

OWNER_EMAIL = os.getenv('ADMIN_OWNER_EMAIL', 'fatsyram@gmail.com').strip().lower()

def database():
    from firebase_admin import firestore
    return firestore.client()

def require_admin(request: Request):
    from firebase_admin import auth
    header = request.headers.get('authorization', '')
    if not header.lower().startswith('bearer '):
        raise HTTPException(401, 'Sign in to access administration.')
    try:
        identity = auth.verify_id_token(header[7:], check_revoked=True)
    except Exception:
        raise HTTPException(401, 'Your session expired. Sign in again.')
    if not identity.get('email_verified'):
        raise HTTPException(403, 'Verify your email before accessing administration.')
    uid = identity['uid']
    email = identity.get('email', '').lower()
    ref = database().collection('adminAccess').document(uid)
    if email == OWNER_EMAIL:
        if not ref.get().exists:
            ref.set({'email': email, 'role': 'owner', 'active': True, 'updatedAt': datetime.now(timezone.utc)})
        return {'uid': uid, 'email': email, 'role': 'owner'}
    role = ref.get().to_dict() or {}
    if role.get('active') is not True or role.get('role') != 'admin':
        raise HTTPException(403, 'Admin access is required.')
    return {'uid': uid, 'email': email, 'role': 'admin'}

def require_owner(identity=Depends(require_admin)):
    if identity['role'] != 'owner':
        raise HTTPException(403, 'Only the owner can manage admins.')
    return identity

def serialize(value):
    if isinstance(value, datetime): return value.isoformat()
    if isinstance(value, dict): return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, list): return [serialize(v) for v in value]
    return value

def audit(actor, action, target):
    database().collection('adminAudit').add({'actor': actor['uid'], 'action': action, 'target': target, 'createdAt': datetime.now(timezone.utc)})

router = APIRouter(prefix='/admin/workspace', dependencies=[Depends(require_admin)])

@router.get('/me')
def me(identity=Depends(require_admin)):
    return identity

@router.get('/users')
def users(cursor: str = '', identity=Depends(require_admin)):
    q = database().collection('users').order_by('__name__').limit(101)
    if cursor:
        q = q.start_after({ '__name__': database().collection('users').document(cursor) })
    docs = list(q.stream())
    items = []
    for doc in docs[:100]:
        data = doc.to_dict() or {}
        items.append({'uid': doc.id, 'email': data.get('email', ''), 'name': data.get('displayName') or data.get('name', ''),
                      'usage': data.get('usage', {}), 'createdAt': data.get('createdAt'),
                      'lastActive': data.get('lastActiveAt') or data.get('lastLoginAt') or (data.get('progress') or {}).get('lastLoginDate'),
                      'onboarding': {'examDate': (data.get('onboarding') or {}).get('examDate')}})
    return serialize({'items': items, 'cursor': docs[99].id if len(docs) > 100 else None})

@router.get('/users/{uid}')
def user_detail(uid: str):
    db = database()
    user = db.collection('users').document(uid).get()
    if not user.exists: raise HTTPException(404, 'User not found.')
    chats = list(db.collection('chats').where('userId', '==', uid).limit(101).stream())
    exams = list(db.collection('users').document(uid).collection('exams').limit(101).stream())
    return serialize({'exams': [{'id': d.id, **d.to_dict()} for d in exams[:100]],
                      'chats': [{'id': d.id, **{k: d.to_dict().get(k) for k in ['title','updatedAt','createdAt']}} for d in chats[:100]],
                      'truncated': len(chats) > 100 or len(exams) > 100})

class Grant(BaseModel):
    email: str = Field(min_length=3, max_length=254)

@router.get('/admins')
def admins(identity=Depends(require_owner)):
    return serialize({'items': [{'uid': d.id, **d.to_dict()} for d in database().collection('adminAccess').stream()]})

@router.post('/admins')
def grant(body: Grant, identity=Depends(require_owner)):
    from firebase_admin import auth
    try: user = auth.get_user_by_email(body.email.strip().lower())
    except auth.UserNotFoundError: raise HTTPException(404, 'This email must have an existing account.')
    if user.email.lower() == OWNER_EMAIL: raise HTTPException(400, 'The owner already has permanent access.')
    database().collection('adminAccess').document(user.uid).set({'email': user.email, 'role': 'admin', 'active': True, 'updatedAt': datetime.now(timezone.utc), 'updatedBy': identity['uid']})
    audit(identity, 'grant_admin', user.uid)
    return {'status': 'granted'}

@router.delete('/admins/{uid}')
def revoke(uid: str, identity=Depends(require_owner)):
    ref = database().collection('adminAccess').document(uid)
    data = ref.get().to_dict() or {}
    if uid == identity['uid'] or data.get('role') == 'owner' or data.get('email', '').lower() == OWNER_EMAIL:
        raise HTTPException(400, 'Owner access cannot be revoked.')
    ref.set({'active': False, 'updatedBy': identity['uid'], 'updatedAt': datetime.now(timezone.utc)}, merge=True)
    audit(identity, 'revoke_admin', uid)
    return {'status': 'revoked'}

class Draft(BaseModel):
    uid: str = Field(default='', max_length=128, pattern=r'^[^/]*$')
    email: str = Field(default='', max_length=254)
    subject: str = Field(min_length=1, max_length=180)
    message: str = Field(min_length=1, max_length=10000)
    html: str = Field(default='', max_length=50000)

class AIDraft(BaseModel):
    rewrite_copy: bool = False
    instructions: str = Field(min_length=1, max_length=3000)
    subject: str = Field(default='', max_length=180)
    message: str = Field(default='', max_length=10000)
    html: str = Field(default='', max_length=50000)
    audience: Literal['one','selected','all','pro','free'] = 'one'

class LayoutPreview(BaseModel):
    subject: str = Field(default='', max_length=180)
    message: str = Field(default='', max_length=10000)
    html: str = Field(default='', max_length=50000)

@router.post('/email/layout-preview')
def layout_preview(body: LayoutPreview):
    from services import email_sender
    return {'previewOnly': True, 'subject': body.subject.strip() or '(No subject yet)',
            'to': 'Recipient not confirmed', 'from': email_sender._from_address(),
            'html': (sanitize_email_html(body.html) if body.html.strip() else personal_email_html(body.message.strip() or 'Your message will appear here.')) + email_sender._footer('preview-only')}

@router.post('/email/ai-draft')
async def ai_draft(body: AIDraft):
    from services.admin_email_ai import draft_email
    if not body.instructions.strip(): raise HTTPException(400, 'Describe the email you want to write.')
    try:
        return await draft_email(body.instructions, body.subject, body.message, body.audience, body.html, body.rewrite_copy)
    except Exception:
        raise HTTPException(502, 'Could not generate a draft right now. Your current email is unchanged. Please try again.')

@router.post('/email/preview')
def preview(body: Draft, identity=Depends(require_admin)):
    from firebase_admin import auth
    from services import email_sender
    if not body.subject.strip() or not body.message.strip(): raise HTTPException(400, 'Enter a subject and message.')
    if not body.uid and not body.email.strip(): raise HTTPException(400, 'Enter a recipient email.')
    try: recipient = auth.get_user(body.uid) if body.uid else auth.get_user_by_email(body.email.strip().lower())
    except auth.UserNotFoundError: raise HTTPException(404, 'Recipient not found.')
    except ValueError: raise HTTPException(400, 'Enter a valid recipient email.')
    if not recipient.email or recipient.disabled: raise HTTPException(400, 'Recipient is unavailable.')
    reason = email_sender.is_suppressed(database(), recipient.uid)
    if reason: raise HTTPException(400, 'Email suppressed: ' + reason)
    draft_id = str(uuid4())
    data = {'actor': identity['uid'], 'uid': recipient.uid, 'to': recipient.email, 'subject': body.subject.strip(),
            'html': sanitize_email_html(body.html), 'message': body.message.strip(), 'status': 'draft', 'createdAt': datetime.now(timezone.utc)}
    database().collection('adminMail').document(draft_id).create(data)
    return serialize({'id': draft_id, **data, 'sendingEnabled': email_sender.is_enabled(),
                      'from': email_sender._from_address(),
                      'html': (data.get('html') or personal_email_html(data['message'])) + email_sender._footer(recipient.uid)})

def personal_email_html(message):
    paragraphs = ''.join('<p style="margin:0 0 18px">' + html.escape(p).replace('\n', '<br>') + '</p>'
                         for p in message.split('\n\n'))
    return ('<div style="max-width:560px;margin:32px auto;padding:24px;font:16px Arial,sans-serif;'
            'line-height:1.7;color:#292524;overflow-wrap:anywhere">'
            '<div style="font-size:14px;font-weight:bold;color:#bc6a58;margin-bottom:28px">NurseQuizAI</div>'
            + paragraphs + '</div>')

class CampaignDraft(Draft):
    audience: Literal['selected', 'all', 'pro', 'free']
    emails: list[str] = Field(default_factory=list, max_length=100)

@router.post('/email/campaign/preview')
def campaign_preview(body: CampaignDraft, identity=Depends(require_admin)):
    from firebase_admin import auth
    from services import email_sender
    import json
    if not body.subject.strip() or not body.message.strip():
        raise HTTPException(400, 'Enter a subject and message.')
    db = database()
    if body.audience == 'selected':
        addresses = list(dict.fromkeys(e.strip().lower() for e in body.emails if e.strip()))
        if not addresses: raise HTTPException(400, 'Enter at least one student email.')
        candidates = []
        for address in addresses:
            try: user = auth.get_user_by_email(address)
            except (auth.UserNotFoundError, ValueError):
                raise HTTPException(400, 'No existing account for: ' + address)
            candidates.append(user)
    else:
        docs = list(db.collection('users').limit(5001).stream())
        if len(docs) > 5000: raise HTTPException(400, 'Audience exceeds 5,000 accounts. Choose a smaller audience.')
        ids = [d.id for d in docs if body.audience == 'all' or (d.to_dict().get('usage') or {}).get('tier', 'free') == body.audience]
        candidates = []
        for offset in range(0, len(ids), 100):
            candidates.extend(auth.get_users([auth.UidIdentifier(uid) for uid in ids[offset:offset+100]]).users)
    recipients, seen, excluded = [], set(), 0
    for user in candidates:
        address = (user.email or '').strip().lower()
        if not address or user.disabled or address in seen or email_sender.is_suppressed(db, user.uid):
            excluded += 1
            continue
        seen.add(address)
        recipients.append({'uid': user.uid, 'to': user.email, 'status': 'pending'})
    if not recipients: raise HTTPException(400, 'No eligible recipients in this audience.')
    if len(json.dumps(recipients).encode()) > 750000: raise HTTPException(400, 'Audience is too large. Choose a smaller group.')
    draft_id = str(uuid4())
    data = {'kind': 'campaign', 'actor': identity['uid'], 'audience': body.audience,
            'subject': body.subject.strip(), 'html': sanitize_email_html(body.html), 'message': body.message.strip(), 'recipients': recipients,
            'to': str(len(recipients)) + ' students', 'total': len(recipients), 'excluded': excluded,
            'status': 'draft', 'createdAt': datetime.now(timezone.utc), 'sendingEnabled': email_sender.is_enabled()}
    db.collection('adminMail').document(draft_id).create(data)
    return serialize({'id': draft_id, **data, 'from': email_sender._from_address(),
                      'remainingToday': email_sender.remaining_today(db),
                      'html': (data.get('html') or personal_email_html(data['message'])) + email_sender._footer(recipients[0]['uid'])})

@router.post('/email/{draft_id}/batch')
def campaign_send(draft_id: str, identity=Depends(require_admin)):
    from firebase_admin import firestore, auth
    from services import email_sender
    db = database()
    ref = db.collection('adminMail').document(draft_id)
    @firestore.transactional
    def reserve(tx):
        data = ref.get(transaction=tx).to_dict() or {}
        if data.get('actor') != identity['uid']: raise HTTPException(403, 'This campaign belongs to another admin.')
        if data.get('kind') != 'campaign' or data.get('status') not in ('draft', 'paused'):
            raise HTTPException(409, 'Campaign is already processing or finished. Refresh history.')
        if data.get('sendingEnabled') != email_sender.is_enabled():
            raise HTTPException(409, 'Sending mode changed. Create a new preview.')
        if data['status'] == 'draft' and (datetime.now(timezone.utc)-data['createdAt']).total_seconds() > 3600:
            raise HTTPException(409, 'Preview expired. Create a new preview.')
        tx.update(ref, {'status': 'sending'})
        return data
    data = reserve(db.transaction())
    recipients = data['recipients']
    reason = ''
    try:
        for recipient in [r for r in recipients if r['status'] == 'pending'][:5]:
            if email_sender.remaining_today(db) <= 0:
                reason = 'Daily sending limit reached. Resume when the daily allowance resets.'
                break
            try: user = auth.get_user(recipient['uid'])
            except auth.UserNotFoundError: user = None
            if not user or user.disabled or user.email != recipient['to']:
                recipient.update(status='skipped', reason='Recipient account changed or is unavailable.')
            else:
                result = email_sender.send_email(db, uid=recipient['uid'], to=recipient['to'], subject=data['subject'],
                    html=(data.get('html') or personal_email_html(data['message'])), campaign='admin_announcement',
                    idempotency_key='admin_' + draft_id + '_' + recipient['uid'])
                if result.get('reason', '').startswith('daily_cap_reached'):
                    reason = 'Daily sending limit reached. Resume later.'
                    break
                recipient.update(status=result['status'], reason=result.get('reason', ''))
            ref.update({'recipients': recipients})
        pending = sum(r['status'] == 'pending' for r in recipients)
        counts = {status: sum(r['status'] == status for r in recipients) for status in ('sent','dry_run','skipped','failed','pending')}
        result = {'status': 'paused' if pending else 'completed', 'counts': counts, 'reason': reason, 'canContinue': bool(pending and not reason)}
        ref.update({'status': result['status'], 'result': result, 'updatedAt': datetime.now(timezone.utc)})
    except Exception:
        raise HTTPException(503, 'Campaign outcome is uncertain. Check history before sending again.')
    audit(identity, 'send_campaign_batch', draft_id)
    return result

@router.post('/email/{draft_id}/send')
def send(draft_id: str, identity=Depends(require_admin)):
    from firebase_admin import firestore
    from services import email_sender
    db = database()
    ref = db.collection('adminMail').document(draft_id)
    @firestore.transactional
    def reserve(tx):
        snap = ref.get(transaction=tx)
        data = snap.to_dict() or {}
        if data.get('actor') != identity['uid']: raise HTTPException(403, 'This draft belongs to another admin.')
        if data.get('kind') == 'campaign': raise HTTPException(400, 'Use campaign sending for this draft.')
        if data.get('status') != 'draft': raise HTTPException(409, 'This send was already requested. Check its history before sending again.')
        if (datetime.now(timezone.utc) - data['createdAt']).total_seconds() > 3600: raise HTTPException(409, 'Preview expired. Create a new preview.')
        tx.update(ref, {'status': 'sending'})
        return data
    data = reserve(db.transaction())
    try:
        result = email_sender.send_email(db, uid=data['uid'], to=data['to'], subject=data['subject'],
            html=(data.get('html') or personal_email_html(data['message'])),
            campaign='admin_personal', idempotency_key='admin_' + draft_id)
        ref.update({'status': result['status'], 'result': result, 'sentAt': datetime.now(timezone.utc)})
    except Exception:
        # Never automatically retry an ambiguous provider result.
        raise HTTPException(503, 'Send outcome is uncertain. Check email history before trying again.')
    audit(identity, 'send_email', draft_id)
    return result

@router.get('/email')
def history():
    docs = database().collection('adminMail').order_by('createdAt', direction='DESCENDING').limit(100).stream()
    return serialize({'items': [{'id': d.id, **d.to_dict()} for d in docs]})

@router.get('/exams')
def upcoming_exams(days: int = 30):
    from datetime import timedelta
    if days not in (7, 14, 30, 90): raise HTTPException(400, 'Choose 7, 14, 30 or 90 days.')
    db = database()
    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=days)
    exam_docs = list(db.collection_group('exams').limit(2001).stream())
    user_docs = list(db.collection('users').limit(5001).stream())
    user_map = {d.id: d.to_dict() for d in user_docs[:5000]}
    items, seen = [], set()
    def append(uid, key, name, raw, source):
        try:
            date = raw.date() if isinstance(raw, datetime) else datetime.fromisoformat(str(raw).replace('Z', '+00:00')).date()
        except (TypeError, ValueError): return
        if not today <= date <= end or (uid, str(date), name) in seen: return
        seen.add((uid, str(date), name))
        user = user_map.get(uid, {})
        items.append({'uid': uid, 'id': key, 'name': name, 'date': str(date), 'source': source,
                      'email': user.get('email', ''), 'usage': user.get('usage', {}),
                      'lastActive': user.get('lastActiveAt') or user.get('lastLoginAt') or (user.get('progress') or {}).get('lastLoginDate')})
    for doc in exam_docs[:2000]:
        parts = doc.reference.path.split('/')
        if len(parts) != 4 or parts[0] != 'users': continue
        data = doc.to_dict()
        append(parts[1], doc.id, data.get('name') or data.get('subject') or 'Exam', data.get('date'), 'Exam calendar')
    for uid, user in user_map.items():
        onboarding = user.get('onboarding') or {}
        if not any(item['uid'] == uid for item in items):
            append(uid, 'onboarding', onboarding.get('examType') or 'Exam', onboarding.get('examDate'), 'Onboarding')
    return serialize({'items': sorted(items, key=lambda x: x['date']), 'truncated': len(exam_docs) > 2000 or len(user_docs) > 5000})
