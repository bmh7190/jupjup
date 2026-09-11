from __future__ import annotations

import base64
import unittest
from datetime import date
from unittest.mock import Mock, patch

from jupjup.models import LostItemQuery, RecordSource, SearchRecord, VisionAssessment
from jupjup.vision import ImageDownloadError, VisionMatcher, _download_image_as_data_url


class FakeHeaders:
    def __init__(self, content_type: str, content_length: int) -> None:
        self.content_type = content_type
        self.content_length = content_length

    def get_content_type(self) -> str:
        return self.content_type

    def get(self, name: str) -> str | None:
        if name.lower() == "content-length":
            return str(self.content_length)
        return None


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "image/jpeg") -> None:
        self.body = body
        self.headers = FakeHeaders(content_type, len(body))

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.body[:size]


def found_record(image_url: str) -> SearchRecord:
    return SearchRecord(
        source=RecordSource.POLICE_FOUND,
        record_type="found",
        atc_id="F1",
        item_name="검은색 지갑",
        image_url=image_url,
    )


class VisionMatcherTest(unittest.TestCase):
    @patch("jupjup.vision.urllib.request.urlopen")
    def test_police_image_is_converted_to_data_url(self, urlopen: Mock) -> None:
        image_bytes = b"small-jpeg"
        urlopen.return_value = FakeResponse(image_bytes)

        result = _download_image_as_data_url(
            "https://minwon24.police.go.kr/lost112/find/image.do"
        )

        self.assertEqual(
            result,
            "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"),
        )

    def test_unapproved_image_host_is_rejected(self) -> None:
        with self.assertRaises(ImageDownloadError):
            _download_image_as_data_url("https://example.com/image.jpg")

    @patch("jupjup.vision._download_image_as_data_url")
    def test_image_download_failure_skips_only_vision_score(self, download: Mock) -> None:
        download.side_effect = ImageDownloadError("timeout")
        matcher = object.__new__(VisionMatcher)
        matcher._model = Mock()

        result = matcher.score(
            LostItemQuery(item_name="지갑", lost_date=date(2026, 9, 11)),
            found_record("https://minwon24.police.go.kr/image.jpg"),
        )

        self.assertIsNone(result)
        matcher._model.invoke.assert_not_called()

    @patch("jupjup.vision._download_image_as_data_url")
    def test_vision_model_receives_downloaded_data_url(self, download: Mock) -> None:
        download.return_value = "data:image/jpeg;base64,YWJj"
        matcher = object.__new__(VisionMatcher)
        matcher._model = Mock(
            invoke=Mock(return_value=VisionAssessment(similarity=0.8, reason="색상 일치"))
        )

        result = matcher.score(
            LostItemQuery(item_name="지갑"),
            found_record("https://minwon24.police.go.kr/image.jpg"),
        )

        self.assertEqual(result.similarity, 0.8)  # type: ignore[union-attr]
        message = matcher._model.invoke.call_args.args[0][0]
        self.assertEqual(message.content[1]["image_url"]["url"], download.return_value)


if __name__ == "__main__":
    unittest.main()
