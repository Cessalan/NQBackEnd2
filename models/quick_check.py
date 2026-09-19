"""Versioned counterparts of the frontend quick-check records.

Recompute grading from the saved selection/key. These are client-origin
questions: validation establishes consistency, not server-issued provenance.
"""
from datetime import datetime
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, StrictInt, StrictBool, model_validator


def topic_key(value: str) -> str:
    return ' '.join(value.split()).lower()


class QuestionAnswerRecord(BaseModel):
    schemaVersion: Literal[1]
    checkId: str
    questionId: str
    questionIndex: StrictInt
    source: Literal['quick_check']
    attempt: Literal[1]
    recordedAt: datetime
    topic: str = Field(min_length=1, max_length=500)
    topicKey: str
    concept: Optional[str] = None
    question: str = Field(min_length=1, max_length=10000)
    options: List[str] = Field(min_length=4, max_length=6)
    scenario: Optional[str] = None
    format: Literal['mcq', 'sata', 'casestudy']
    kind: Optional[str] = None
    difficulty: Optional[Union[StrictInt, str]] = None
    correctIndices: List[StrictInt]
    rationale: Optional[str] = None
    selection: Union[StrictInt, List[StrictInt], Literal['unsure']]
    correct: StrictBool
    partial: StrictBool
    unsure: StrictBool
    questionFingerprint: str

    @model_validator(mode='after')
    def validate_answer(self):
        if self.questionIndex < 0 or self.questionId != f'{self.checkId}:{self.questionIndex}':
            raise ValueError('Invalid question identity')
        if not self.topic.strip() or self.topicKey != topic_key(self.topic):
            raise ValueError('Topic identity mismatch')
        if any(not o.strip() for o in self.options) or len({topic_key(o) for o in self.options}) != len(self.options):
            raise ValueError('Options must be nonempty and distinct')
        key = self.correctIndices
        if len(set(key)) != len(key) or any(i < 0 or i >= len(self.options) for i in key):
            raise ValueError('Invalid answer key')
        if self.format == 'sata':
            if not 2 <= len(key) < len(self.options):
                raise ValueError('Invalid select-all key')
        elif len(key) != 1 or len(self.options) != 4:
            raise ValueError('Invalid single-answer key')
        if self.format == 'casestudy' and not (self.scenario or '').strip():
            raise ValueError('Case study requires a scenario')
        unsure = self.selection == 'unsure'
        if unsure:
            correct, partial = False, False
        else:
            if self.format == 'sata':
                if not isinstance(self.selection, list) or not self.selection:
                    raise ValueError('Select-all requires selected indices')
                chosen = self.selection
            else:
                if type(self.selection) is not int:
                    raise ValueError('Single-answer requires one index')
                chosen = [self.selection]
            if len(set(chosen)) != len(chosen) or any(i < 0 or i >= len(self.options) for i in chosen):
                raise ValueError('Invalid selection')
            correct = set(chosen) == set(key)
            partial = self.format == 'sata' and not correct and bool(set(chosen) & set(key))
        if (self.correct, self.partial, self.unsure) != (correct, partial, unsure):
            raise ValueError('Saved grade does not match selection')
        return self


class QuickCheckRecord(BaseModel):
    schemaVersion: Literal[1]
    checkId: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    chatId: str
    funnelId: Optional[str] = None
    completedAt: datetime
    offered: StrictInt = Field(ge=1, le=8)
    answered: StrictInt = Field(ge=1, le=8)
    completion: Literal['completed', 'ended_early']
    answers: List[QuestionAnswerRecord] = Field(min_length=1, max_length=8)
    # Incoming summaries are never used for grading or recommendation.
    topics: list = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_check(self):
        if self.answered != len(self.answers) or self.answered > self.offered:
            raise ValueError('Answer count mismatch')
        expected = 'completed' if self.answered == self.offered else 'ended_early'
        if self.completion != expected:
            raise ValueError('Completion mismatch')
        if any(a.checkId != self.checkId for a in self.answers):
            raise ValueError('Answer belongs to another check')
        if [a.questionIndex for a in self.answers] != list(range(self.answered)):
            raise ValueError('Questions must be unique and in check order')
        return self

    def topic_results(self):
        topics = {}
        for answer in self.answers:
            row = topics.setdefault(answer.topicKey, dict(topicKey=answer.topicKey, topic=answer.topic,
                                                         correct=0, answered=0, partial=0, unsure=0))
            for field in ('correct', 'partial', 'unsure'):
                row[field] += int(getattr(answer, field))
            row['answered'] += 1
        return topics

    def diagnostic(self):
        return {row['topic']: 100 * row['correct'] / row['answered'] for row in self.topic_results().values()}
