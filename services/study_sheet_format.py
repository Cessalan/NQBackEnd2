"""Validate the streamed, versioned sheet before calling it complete."""
import json
import re

VERSION = 2


def _string(value, limit=16000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Invalid study sheet text")
    return value.strip()


def validate_block(raw, source_ids, sources=None):
    if not isinstance(raw, dict):
        raise ValueError("Invalid study sheet block")
    kind = raw.get("kind")
    block = {"kind": kind}
    refs = raw.get("sourceIds") or []
    if not isinstance(refs, list) or any(ref not in source_ids for ref in refs):
        raise ValueError("Unknown study sheet source")
    block["sourceIds"] = list(dict.fromkeys(refs))
    if raw.get('supplemental') is True:
        if refs:
            raise ValueError('Supplemental explanation cannot claim a supplied source')
        block['supplemental'] = True
    if any(r.startswith('P') for r in refs) and not (kind == 'callout' and raw.get('tone') == 'teacher'):
        raise ValueError('Student preferences cannot support academic facts')
    if any(r.startswith('Q') for r in refs) and not (kind == 'callout' and raw.get('tone') == 'practice'):
        raise ValueError('Quiz feedback must be attributed as practice review')
    if raw.get("title"):
        block["title"] = _string(raw["title"], 240)
    if kind in ("paragraph", "callout"):
        block["text"] = _string(raw.get("text"))
        if kind == "callout":
            tone = raw.get("tone", "takeaway")
            if tone not in ("teacher", "practice", "takeaway", "example"):
                raise ValueError("Invalid callout tone")
            if tone in ("teacher", "practice") and not refs:
                raise ValueError("Personalized callout needs evidence")
            if tone == "practice" and not any(s.startswith("Q") for s in refs):
                raise ValueError("Practice callout needs quiz evidence")
            if tone == 'practice' and sources is not None and not any(
                sources[s].get('kind') == 'quiz' and sources[s].get('answered', 0) > 0 for s in refs):
                raise ValueError('Practice callout needs an answered quiz')
            block["tone"] = tone
    elif kind == "list":
        items = raw.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= 60:
            raise ValueError("Invalid study sheet list")
        block.update(items=[_string(i) for i in items], ordered=raw.get("ordered") is True)
    elif kind == "table":
        columns, rows = raw.get("columns"), raw.get("rows")
        if not isinstance(columns, list) or not 2 <= len(columns) <= 5 or not isinstance(rows, list) or not 1 <= len(rows) <= 60:
            raise ValueError("Invalid study sheet table")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
            raise ValueError("Invalid study sheet table row")
        block.update(columns=[_string(c, 200) for c in columns],
                     rows=[[_string(c, 3000) for c in row] for row in rows])
    elif kind == "self_check":
        questions = raw.get("questions")
        if not isinstance(questions, list) or not 1 <= len(questions) <= 15:
            raise ValueError("Invalid study sheet self check")
        block["questions"] = [{"question": _string(q.get("question"), 2000),
                               "answer": _string(q.get("answer"), 5000)} for q in questions if isinstance(q, dict)]
        if len(block["questions"]) != len(questions):
            raise ValueError("Invalid study sheet question")
    else:
        raise ValueError("Unsupported study sheet block")
    return block


class SheetStreamParser:
    """JSON records: header, complete sections, then end (whitespace allowed).

    Only validated sections reach the UI. Missing end or truncated JSON
    causes a retry and never a green Complete badge.
    """
    def __init__(self, sources, language, opening=None):
        self.sheet = {"version": VERSION, "language": language, "sections": [], "sources": sources}
        self.source_ids = {s["id"] for s in sources}
        self.sources = {s['id']: s for s in sources}
        self.buffer = ""
        self.header = False
        self.ended = False
        self.wrapped = False
        self.wrapper_ended = False
        self.expect_comma = False
        self.generated_sections = 0
        self.opening = {**opening, 'blocks': [validate_block(b, self.source_ids, self.sources)
                         for b in opening['blocks']]} if opening else None

    def feed(self, text):
        self.buffer += text
        if len(self.buffer) > 200000:
            raise ValueError("Study sheet record too large")
        events = []
        decoder = json.JSONDecoder()
        while self.buffer.strip():
            self.buffer = self.buffer.lstrip()
            if not self.header and not self.wrapped:
                prefix = re.match(r'^\{\s*"records"\s*:\s*\[', self.buffer)
                if prefix:
                    self.wrapped = True
                    self.buffer = self.buffer[prefix.end():].lstrip()
            if self.wrapped:
                if self.wrapper_ended:
                    raise ValueError('Content after study sheet envelope')
                if self.buffer.startswith(']'):
                    if not self.expect_comma:
                        raise ValueError('Empty or trailing-comma study sheet envelope')
                    suffix = re.match(r'^\]\s*\}', self.buffer)
                    if not suffix:
                        break
                    self.buffer = self.buffer[suffix.end():]
                    self.wrapper_ended = True
                    continue
                if self.expect_comma:
                    if not self.buffer:
                        break
                    if not self.buffer.startswith(','):
                        raise ValueError('Invalid study sheet record separator')
                    self.buffer = self.buffer[1:].lstrip()
                    self.expect_comma = False
            try:
                raw, end = decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                break  # A provider may pretty-print a complete record.
            self.buffer = self.buffer[end:]
            events += self._record(raw)
            self.expect_comma = self.wrapped
        return events

    def _record(self, raw):
        if self.ended or not isinstance(raw, dict):
            raise ValueError("Invalid study sheet record order")
        kind = raw.get("type")
        if kind == "header" and not self.header:
            self.sheet.update(title=_string(raw.get("title"), 160),
                              subtitle=_string(raw.get("subtitle"), 800), summary=_string(raw.get("summary"), 2400))
            self.header = True
            events = [{"status": "study_sheet_header", "studySheet": {**self.sheet, 'sections': []}}]
            if self.opening:
                self.sheet['sections'].append(self.opening)
                events.append({'status': 'study_sheet_section', 'section': self.opening})
            return events
        if kind == "section" and self.header:
            blocks = raw.get("blocks")
            if not isinstance(blocks, list) or not 1 <= len(blocks) <= 40 or len(self.sheet["sections"]) >= 30:
                raise ValueError("Invalid study sheet section")
            section = {"id": f"section-{len(self.sheet['sections']) + 1}",
                       "title": _string(raw.get("title"), 240),
                       "blocks": [validate_block(b, self.source_ids, self.sources) for b in blocks]}
            self.sheet["sections"].append(section)
            self.generated_sections += 1
            return [{"status": "study_sheet_section", "section": section}]
        if kind == "end" and self.header and self.generated_sections:
            self.ended = True
            return []
        raise ValueError("Invalid study sheet record order")

    def finish(self):
        if self.buffer.strip():
            raise ValueError('Invalid or incomplete study sheet JSON')
        if not self.ended or (self.wrapped and not self.wrapper_ended):
            raise ValueError("Study sheet ended before completion")
        return self.sheet


def sheet_to_text(sheet):
    """Readable history for follow-up edits and older clients."""
    lines = [f"# {sheet['title']}", sheet["subtitle"], sheet["summary"]]
    for section in sheet["sections"]:
        lines += ["", f"## {section['title']}"]
        for b in section["blocks"]:
            if b.get("title"):
                lines.append(f"### {b['title']}")
            if b["kind"] in ("paragraph", "callout"):
                lines.append(b["text"])
            elif b["kind"] == "list":
                lines += [f"{i + 1}. {text}" if b["ordered"] else f"- {text}" for i, text in enumerate(b["items"])]
            elif b["kind"] == "table":
                lines += ["| " + " | ".join(b["columns"]) + " |", "| " + " | ".join(["---"] * len(b["columns"])) + " |"]
                lines += ["| " + " | ".join(row) + " |" for row in b["rows"]]
            elif b["kind"] == "self_check":
                lines += [f"{q['question']}\n{q['answer']}" for q in b["questions"]]
    return "\n\n".join(lines)
