"""환경변수와 실행 설정을 한곳에서 관리한다."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


def load_env_file(path: Path | None = None) -> Path | None:
    """python-dotenv가 없어도 사용할 수 있는 작은 .env 로더."""
    candidates = [
        path,
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path.cwd().parent / "outputs" / ".env",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # 로컬 실행에서는 선택한 .env가 셸에 남은 오래된 키보다 우선한다.
            os.environ[key.strip()] = value.strip().strip("'\"")
        return candidate
    return None


@dataclass(frozen=True)
class Settings:
    data_service_key: str
    openai_api_key: str | None
    openai_model: str
    page_size: int
    timeout_seconds: float
    detail_limit: int
    max_pages_per_window: int
    search_budget_seconds: float

    @property
    def api_client_options(self) -> dict[str, int | float]:
        return {
            "page_size": self.page_size,
            "timeout_seconds": self.timeout_seconds,
            "detail_limit": self.detail_limit,
            "max_pages_per_window": self.max_pages_per_window,
            "search_budget_seconds": self.search_budget_seconds,
        }

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Settings":
        load_env_file(env_path)
        raw_service_key = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()
        raw_openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        raw_openai_model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
        if not raw_service_key:
            raise ValueError("DATA_GO_KR_SERVICE_KEY가 없습니다. .env 파일을 확인하세요.")

        return cls(
            # Encoding 키와 Decoding 키를 동일하게 처리한 뒤 요청 시 한 번만 인코딩한다.
            data_service_key=unquote(raw_service_key),
            # Vercel 대시보드에서 붙여 넣을 때 생긴 줄바꿈이 인증 헤더에
            # 포함되지 않도록 외부 환경변수도 파일 입력과 동일하게 정리한다.
            openai_api_key=raw_openai_key or None,
            openai_model=raw_openai_model or "gpt-5.4-mini",
            page_size=max(1, min(int(os.getenv("LOST112_PAGE_SIZE", "10")), 100)),
            timeout_seconds=max(1.0, float(os.getenv("LOST112_TIMEOUT_SECONDS", "15"))),
            detail_limit=max(0, int(os.getenv("LOST112_DETAIL_LIMIT", "5"))),
            max_pages_per_window=max(
                1, int(os.getenv("LOST112_MAX_PAGES_PER_WINDOW", "10"))
            ),
            search_budget_seconds=max(
                0.01, float(os.getenv("LOST112_SEARCH_BUDGET_SECONDS", "30"))
            ),
        )
