"""이전 import 경로를 유지하는 개인정보 마스킹 호환 모듈."""

from .domain.privacy import PATTERNS, mask_pii

__all__ = ["PATTERNS", "mask_pii"]
