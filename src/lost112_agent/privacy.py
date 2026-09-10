"""LLM과 로그로 전달되기 전 명확한 개인정보 패턴을 가린다."""

from __future__ import annotations

import re


PATTERNS = (
    (re.compile(r"\b\d{6}-?[1-4]\d{6}\b"), "[주민등록번호 마스킹]"),
    (re.compile(r"\b01[016789]-?\d{3,4}-?\d{4}\b"), "[전화번호 마스킹]"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"), "[카드번호 마스킹]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[이메일 마스킹]"),
)


def mask_pii(text: str) -> str:
    masked = text
    for pattern, replacement in PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked

