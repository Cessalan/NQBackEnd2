"""Luna settings for source-based practice; speed follows server entitlements."""
import logging

logger = logging.getLogger('uvicorn.error.material_model')
MODEL = 'gpt-6-luna'
# 2026-10-08: analysis and figure reading moved to low reasoning. On a real
# 4-page handout (tools/time_upload_variants.py) analysis went 15.9s -> 9.3s
# with the same coverage (29 -> 30 learning goals), and a figure 7-10s -> ~5s.
# Changing this policy changes every analysis fingerprint, so each saved
# document is re-analysed once, on its next use.
MODEL_POLICY = {'model': MODEL, 'analysisReasoning': 'low', 'visualReasoning': 'low',
                'planReasoning': 'medium', 'draftReasoning': 'low',
                'verificationReasoning': 'medium'}


def service_tier_for_chat(chat_id):
    """Refresh paid membership for each operation, never from client settings."""
    if not chat_id:
        return 'default'
    try:
        from firebase_admin import firestore
        from services.usage_guard import _resolve_uid
        uid = _resolve_uid(chat_id)
        if uid:
            user = firestore.client().collection('users').document(uid).get().to_dict() or {}
            if (user.get('usage') or {}).get('tier') == 'pro':
                return 'fast'
    except Exception:
        logger.warning('Material membership lookup failed; using standard processing')
    return 'default'


def material_model(*, service_tier='default', reasoning_effort='medium',
                   max_completion_tokens=16384, timeout=90):
    from langchain_openai import ChatOpenAI
    if service_tier not in ('default', 'fast'):
        raise ValueError('Invalid material processing tier')
    # This path sends JSON and image input without tools, so Chat Completions
    # supports reasoning. Omit sampling parameters and legacy max_tokens.
    return ChatOpenAI(model=MODEL, reasoning_effort=reasoning_effort,
        service_tier=service_tier, use_responses_api=False,
        max_completion_tokens=max_completion_tokens,
        request_timeout=timeout).bind(response_format={'type': 'json_object'})


def record_usage(response, *, service_tier, reasoning_effort, elapsed):
    """Log actual billing tier and token usage, without source text or user data."""
    metadata = response.response_metadata or {}
    usage = metadata.get('token_usage') or {}
    details = usage.get('completion_tokens_details') or {}
    logger.info('material_model model=%s requested_tier=%s actual_tier=%s '
                'reasoning=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s seconds=%.3f',
                MODEL, service_tier, metadata.get('service_tier', 'unknown'),
                reasoning_effort, usage.get('prompt_tokens'), usage.get('completion_tokens'),
                details.get('reasoning_tokens'), elapsed)
