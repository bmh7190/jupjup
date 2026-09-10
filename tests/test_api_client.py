from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from datetime import date

from jupjup.api_client import API_DEFINITIONS, Lost112ApiClient
from jupjup.models import RecordSource


FOUND_XML = """
<response><header><resultCode>00</resultCode><resultMsg>NORMAL SERVICE.</resultMsg></header>
<body><totalCount>1</totalCount><items><item>
<atcId>F2026090900000001</atcId><fdSn>1</fdSn><fdPrdtNm>검정 카드지갑</fdPrdtNm>
<prdtClNm>지갑 &gt; 카드지갑</prdtClNm><clrNm>블랙(검정)</clrNm>
<fdYmd>2026-09-09</fdYmd><fdPlace>강남역</fdPlace><depPlace>서울강남경찰서</depPlace>
<fdFilePathImg>https://example.com/wallet.jpg</fdFilePathImg>
</item></items></body></response>
"""


class ApiParsingTest(unittest.TestCase):
    def test_found_record_is_normalized(self) -> None:
        root = ET.fromstring(FOUND_XML)
        item = root.find(".//item")
        assert item is not None
        definition = next(d for d in API_DEFINITIONS if d.source == RecordSource.POLICE_FOUND)

        record = Lost112ApiClient._parse_record(item, definition)

        self.assertEqual(record.item_name, "검정 카드지갑")
        self.assertEqual(record.event_date, date(2026, 9, 9))
        self.assertEqual(record.event_place, "강남역")
        self.assertEqual(record.image_url, "https://example.com/wallet.jpg")
        self.assertIn("ATC_ID=F2026090900000001", record.detail_url or "")

    def test_placeholder_image_is_removed(self) -> None:
        root = ET.fromstring(FOUND_XML.replace(
            "https://example.com/wallet.jpg",
            "https://www.lost112.go.kr/lostnfs/images/sub/img02_no_img.gif",
        ))
        item = root.find(".//item")
        assert item is not None
        definition = next(d for d in API_DEFINITIONS if d.source == RecordSource.POLICE_FOUND)
        self.assertIsNone(Lost112ApiClient._parse_record(item, definition).image_url)


if __name__ == "__main__":
    unittest.main()

