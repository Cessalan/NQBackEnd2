"""Read-only projections for the admin's historical study-plan inspector.

Called only by the authenticated admin router. Never generates content, replays
student actions, or repairs historical records while inspecting them.
"""
from fastapi import HTTPException


def _id(value):
    if not value or '/' in value or len(value) > 1500:
        raise HTTPException(400, 'Invalid record ID.')
    return value


def _fields(data, names):
    return {key: data[key] for key in names if key in data}


def _page(collection, cursor, size, fields=None):
    query = collection.order_by('__name__').limit(size + 1)
    if fields:
        query = query.select(fields)
    if cursor:
        query = query.start_after({'__name__': collection.document(_id(cursor))})
    docs = list(query.stream())
    return docs[:size], docs[size - 1].id if len(docs) > size else None


def list_plans(db, cursor=''):
    collection = db.collection('chats')
    # All existing plans have this flag; no date cutoff, so old plans stay visible.
    query = collection.where('isStudySession', '==', True).order_by('__name__').limit(51)
    query = query.select(['title', 'userId', 'createdAt', 'updatedAt', 'study.status', 'study.startedAt'])
    if cursor:
        query = query.start_after({'__name__': collection.document(_id(cursor))})
    docs = list(query.stream())
    return {'items': [{**doc.to_dict(), 'id': doc.id} for doc in docs[:50]],
            'cursor': docs[49].id if len(docs) > 50 else None}


def plan_detail(db, chat_id):
    ref = db.collection('chats').document(_id(chat_id))
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(404, 'This study session was not found.')
    data = doc.to_dict() or {}
    uid = data.get('userId')
    owner, performance = {}, {}
    if uid:
        user = db.collection('users').document(_id(uid))
        owner = _fields(user.get().to_dict() or {}, ['email', 'displayName', 'name'])
        performance = user.collection('studyPerformance').document(chat_id).get().to_dict() or {}
    return {'chat': {**_fields(data, ['title', 'userId', 'createdAt', 'updatedAt', 'study',
                                     'examDate', 'examName', 'courseContext', 'isStudySession']), 'id': doc.id},
            'owner': owner, 'performance': _fields(performance, ['history', 'topics', 'formats', 'updatedAt']),
            'readOnly': True}


def message_page(db, chat_id, cursor=''):
    ref = db.collection('chats').document(_id(chat_id))
    if not ref.get().exists:
        raise HTTPException(404, 'This study session was not found.')
    docs, next_cursor = _page(ref.collection('messages'), cursor, 100, [
        'id', 'nodeId', 'type', 'role', 'timestamp', 'createdAt', 'hidden', 'content',
        'sourceQuizId', 'quizProgress', 'practice.answers', 'practice.firstAnswers',
        'practice.snapshot.firstAttemptStatuses', 'flashcardProgress',
    ])
    items = []
    for doc in docs:
        data = doc.to_dict() or {}
        item = _fields(data, ['nodeId', 'type', 'role', 'timestamp', 'createdAt', 'hidden', 'sourceQuizId'])
        item.update(id=doc.id, storedId=data.get('id'), preview=str(data.get('content') or '')[:240])
        progress = data.get('quizProgress') or {}
        practice = data.get('practice') or {}
        item['answerRecords'] = {
            'first': progress.get('firstAttemptAnswers') or practice.get('firstAnswers') or {},
            'statuses': progress.get('firstAttemptStatuses') or (practice.get('snapshot') or {}).get('firstAttemptStatuses') or {},
            'latest': progress.get('answers') or practice.get('answers') or {},
        }
        item['progressUpdatedAt'] = progress.get('lastUpdated') or (data.get('flashcardProgress') or {}).get('lastUpdated')
        items.append(item)
    return {'items': items, 'cursor': next_cursor}


def message_detail(db, chat_id, message_id):
    ref = db.collection('chats').document(_id(chat_id)).collection('messages').document(_id(message_id))
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(404, 'This saved step was not found.')
    data = doc.to_dict() or {}
    message = _fields(data, ['nodeId', 'type', 'role', 'timestamp', 'createdAt', 'content', 'hidden',
        'studyContent', 'quizData', 'quizProgress', 'practice', 'flashcardProgress', 'sourceQuizId',
        'reviewQuestion', 'reviewAnswer', 'reviewFeedback', 'practicePrompt', 'questionCount'])
    message.update(id=doc.id, storedId=data.get('id'))
    discussions, discussion_cursor = _page(ref.collection('reasoningDiscussions'), '', 200)
    summaries, summary_cursor = _page(ref.collection('reasoningSummaries'), '', 20)
    return {'message': message,
            'discussions': [{'id': d.id, **_fields(d.to_dict() or {}, ['history', 'createdAt', 'updatedAt'])} for d in discussions],
            'summaries': [{'id': d.id, **_fields(d.to_dict() or {}, ['summaries', 'focus', 'createdAt', 'updatedAt'])} for d in summaries],
            'reasoningTruncated': bool(discussion_cursor or summary_cursor), 'readOnly': True}
