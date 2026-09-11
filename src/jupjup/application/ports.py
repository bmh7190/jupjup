"""애플리케이션 계층이 외부 구현에 요구하는 최소 인터페이스."""

from __future__ import annotations

from typing import Collection, Protocol

from ..domain.models import (
    ApiSearchResponse,
    LostItemQuery,
    RecordSource,
    SearchRecord,
    VisionAssessment,
)


class LostItemSearchPort(Protocol):
    def search_all(
        self,
        query: LostItemQuery,
        *,
        sources: Collection[RecordSource] | None = None,
    ) -> tuple[list[ApiSearchResponse], dict[str, str]]: ...


class VisionMatcherPort(Protocol):
    def score(
        self,
        query: LostItemQuery,
        record: SearchRecord,
    ) -> VisionAssessment | None: ...
