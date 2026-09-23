"""Restricted email markup: inline typography, tables and HTTPS links only."""
import html
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

TAGS = set('div span p h1 h2 h3 h4 strong b em i u br hr ul ol li table tbody thead tr td th a'.split())
PROPERTIES = set(('color background-color font-family font-size font-weight font-style line-height '
                  'text-align text-decoration letter-spacing padding padding-top padding-bottom padding-left padding-right '
                  'margin margin-top margin-bottom margin-left margin-right border border-top border-bottom border-color '
                  'border-radius border-collapse border-spacing width max-width height vertical-align').split())

def clean_style(value):
    rules = []
    for declaration in value.split(';'):
        key, sep, val = declaration.partition(':')
        key, val = key.strip().lower(), val.strip()
        if sep and key in PROPERTIES and re.fullmatch(r'[a-zA-Z0-9#.,%\s\-"\']+', val):
            rules.append(f'{key}:{val}')
    return ';'.join(rules)

class EmailHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output, self.stack = [], []
        self.blocked = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script','style','iframe','object','svg','math'):
            self.blocked += 1
            return
        if self.blocked or tag not in TAGS: return
        safe = []
        for key, value in attrs:
            value = value or ''
            if key == 'style': value = clean_style(value)
            elif tag == 'a' and key == 'href':
                parsed = urlsplit(value)
                if parsed.scheme != 'https' or not parsed.netloc: continue
            elif key in ('width','height','cellpadding','cellspacing','colspan','rowspan'):
                if not re.fullmatch(r'\d{1,4}%?', value): continue
            elif key == 'role' and value == 'presentation': pass
            else: continue
            safe.append(f' {key}="{html.escape(value, quote=True)}"')
        self.output.append('<' + tag + ''.join(safe) + '>')
        if tag not in ('br','hr'): self.stack.append(tag)
    def handle_endtag(self, tag):
        if tag in ('script','style','iframe','object','svg','math'):
            self.blocked = max(0, self.blocked - 1)
            return
        if self.blocked or tag not in self.stack: return
        while self.stack:
            current = self.stack.pop()
            self.output.append('</' + current + '>')
            if current == tag: break
    def handle_data(self, data):
        if not self.blocked: self.output.append(html.escape(data))

def sanitize_email_html(markup):
    parser = EmailHTML()
    parser.feed(markup)
    parser.close()
    return ''.join(parser.output) + ''.join('</' + tag + '>' for tag in reversed(parser.stack))
