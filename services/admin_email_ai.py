"""Generate email designs without access to recipients or sending."""
import json
import re
from pydantic import BaseModel, Field

class EmailSuggestion(BaseModel):
    subject: str = Field(max_length=180)
    message: str = Field(min_length=1, max_length=10000)
    html: str = Field(min_length=1, max_length=50000)

async def draft_email(instructions, subject, message, audience, current_html='', rewrite_copy=False):
    from anthropic import AsyncAnthropic
    async with AsyncAnthropic(timeout=45.0, max_retries=1) as client:
        response = await client.messages.create(
            model='claude-haiku-4-5', max_tokens=6500,
            system='''You help NurseQuizAI admins write emails to nursing students.
Your primary job is VISUAL DESIGN of the supplied copy. Unless rewrite_copy is true,
preserve every word of current_message in the HTML and return current_subject/current_message
unchanged. Do not invent headings or CTA labels: use supplied wording. You may split the
existing text into headings, paragraphs, and cards without changing its words. Design direction
is a visual instruction, not permission to rewrite. If rewrite_copy is true you may improve copy.
Return only JSON with subject, message and html strings. Message is the plain-text equivalent, with blank lines
between short paragraphs. Use readable bullets when helpful. Follow the requested tone, language,
length and structure. Default to warm, concise, natural copy without hype. Revise the existing
draft if supplied. Use only product facts provided by the admin; do not invent features, prices,
links, discounts, deadlines, student names or success claims. Do not add an unsubscribe footer
or postal address: the email service provides those. Never claim you sent an email.
Generate a polished email HTML fragment with INLINE CSS. Design the visual layout requested:
clear heading hierarchy, generous spacing, tasteful colored panels, benefit lists, and a CTA
only if the admin supplied its HTTPS destination. Default to a white background, charcoal text,
coral #e88d7d accents, Arial sans-serif, and a centered fluid 560px layout. Include NurseQuizAI branding.
Use presentation tables for layout and email-compatible inline styles. Use hex colors.
Allowed tags: div span p h1 h2 h3 h4 strong b em i u br hr ul ol li table tbody thead tr td th a.
No scripts, images, forms, external stylesheets, style tags, CSS url(), flex or grid.
Preserve provided product facts and revise current_html when supplied. The service adds the footer.''',
            messages=[{'role':'user','content':json.dumps({'instructions':instructions,
                'current_subject':subject,'current_message':message,'audience':audience,'current_html':current_html,'rewrite_copy':rewrite_copy})}])
    text = ''.join(b.text for b in response.content if getattr(b, 'type', '') == 'text').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    result = EmailSuggestion(**json.loads(text))
    if not result.message.strip():
        raise ValueError('Empty draft')
    from services.email_html import sanitize_email_html
    return {'subject':result.subject.strip() if rewrite_copy else subject,
            'message':result.message.strip() if rewrite_copy else message, 'html':sanitize_email_html(result.html)}
