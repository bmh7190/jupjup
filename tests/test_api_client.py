from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from datetime import date
from unittest.mock import patch

from jupjup.api_client import (
    API_DEFINITIONS,
    Lost112ApiClient,
    _product_category_codes,
)
from jupjup.models import LostItemQuery, RecordSource, SearchRecord


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
    def test_wallet_category_codes_are_resolved(self) -> None:
        self.assertEqual(_product_category_codes("검정 카드지갑"), ("PRH000", None))
        self.assertEqual(_product_category_codes("남성용 지갑"), ("PRH000", "PRH200"))

    def test_lost_search_uses_working_category_operation_and_start_date(self) -> None:
        definition = next(
            d for d in API_DEFINITIONS if d.source == RecordSource.POLICE_LOST
        )
        client = Lost112ApiClient("test-key")

        params = client._build_list_params(
            definition,
            LostItemQuery(item_name="지갑", lost_date=date(2026, 9, 9)),
        )

        self.assertEqual(definition.list_operation, "getLostGoodsInfoAccToClAreaPd")
        self.assertEqual(params["START_YMD"], "20260909")
        self.assertEqual(params["PRDT_CL_CD_01"], "PRH000")
        self.assertNotIn("LST_PRDT_NM", params)

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
        self.assertIn("selectFindListDetail.do", record.detail_url or "")
        self.assertIn("pkupCmdtyMngId=F2026090900000001", record.detail_url or "")
        self.assertIn("sortSn=1", record.detail_url or "")

    def test_lost_report_does_not_link_to_homepage(self) -> None:
        root = ET.fromstring(
            """
            <response><body><items><item>
            <atcId>L2026090900000001</atcId><lstPrdtNm>지갑</lstPrdtNm>
            </item></items></body></response>
            """
        )
        item = root.find(".//item")
        assert item is not None
        definition = next(
            d for d in API_DEFINITIONS if d.source == RecordSource.POLICE_LOST
        )

        self.assertIsNone(Lost112ApiClient._parse_record(item, definition).detail_url)

    def test_placeholder_image_is_removed(self) -> None:
        root = ET.fromstring(FOUND_XML.replace(
            "https://example.com/wallet.jpg",
            "https://www.lost112.go.kr/lostnfs/images/sub/img02_no_img.gif",
        ))
        item = root.find(".//item")
        assert item is not None
        definition = next(d for d in API_DEFINITIONS if d.source == RecordSource.POLICE_FOUND)
        self.assertIsNone(Lost112ApiClient._parse_record(item, definition).image_url)

    def test_detail_timeout_keeps_list_record(self) -> None:
        definition = next(
            d for d in API_DEFINITIONS if d.source == RecordSource.POLICE_FOUND
        )
        client = Lost112ApiClient("test-key")
        record = SearchRecord(
            source=RecordSource.POLICE_FOUND,
            record_type="found",
            atc_id="F1",
            item_name="검정 카드지갑",
        )

        with patch.object(client, "_request_xml", side_effect=TimeoutError):
            result = client._fetch_and_merge_detail(definition, record)

        self.assertEqual(result, record)


if __name__ == "__main__":
    unittest.main()
