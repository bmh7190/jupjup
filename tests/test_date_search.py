from datetime import date
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from unittest.mock import patch

from jupjup.api_client import Lost112ApiClient, API_DEFINITIONS, build_search_windows
from jupjup.models import LostItemQuery
from jupjup.service import JupJupAgentService


def xml(items, total=None):
    body = ''.join(f'<item><atcId>{key}</atcId><fdSn>1</fdSn><fdPrdtNm>{name}</fdPrdtNm><fdYmd>{day}</fdYmd></item>' for key, name, day in items)
    return ET.fromstring(f'<response><header><resultCode>00</resultCode></header><body><totalCount>{len(items) if total is None else total}</totalCount><items>{body}</items></body></response>')


class DateSearchTest(unittest.TestCase):
    def test_today_clips_second_window_and_omits_future_third_window(self):
        self.assertEqual(build_search_windows(date(2026,3,14),date(2026,3,23)),
                         [(date(2026,3,14),date(2026,3,20)), (date(2026,3,21),date(2026,3,23))])

    def test_month_end_and_year_boundary(self):
        self.assertEqual(build_search_windows(date(2025,12,17),date(2026,3,1))[-1],
                         (date(2025,12,31),date(2026,1,30)))
        self.assertEqual(build_search_windows(date(2024,1,17),date(2024,4,1))[-1],
                         (date(2024,1,31),date(2024,2,28)))

    def test_no_date_keeps_name_search(self):
        client=Lost112ApiClient('test',detail_limit=0)
        calls=[]
        def request(url,params):
            calls.append((url,params))
            return xml([])
        with patch.object(client,'_request_xml',side_effect=request):
            client.search_all(LostItemQuery(item_name='지갑'))
        self.assertEqual(len(calls),3)
        # 날짜 없을 때는 날짜 기간 조회(_search_dated) 경로를 타지 않아야 한다.
        # 경찰청 습득물/포털 조회는 START_YMD 없어야 하고, 분실물 조회는 _build_list_params 로 START_YMD 포함 가능.
        found_calls = [(url, params) for url, params in calls if 'LosfundInfo' in url]
        self.assertTrue(all('START_YMD' not in params for _, params in found_calls))

    def test_page_limit_is_reported_as_partial_notice_not_failure(self):
        client=Lost112ApiClient('test',page_size=1,detail_limit=0,max_pages_per_window=2)
        with patch('jupjup.api_client.API_DEFINITIONS',(API_DEFINITIONS[1],)), patch.object(client,'_request_xml',return_value=xml([('wallet','지갑','2026-03-15')],100)):
            responses, errors=client.search_all(LostItemQuery(item_name='지갑',lost_date=date(2026,3,14)))
        self.assertEqual(len(responses[0].records),1)
        self.assertFalse(responses[0].search_scopes[0].complete)
        self.assertEqual(responses[0].search_scopes[0].pages_completed,2)
        self.assertEqual(
            responses[0].search_scopes[0].partial_reason,
            '설정된 페이지 상한에 도달',
        )
        self.assertFalse(errors)

    def test_dated_search_fetches_details_once_after_collecting_windows(self):
        client=Lost112ApiClient('test',page_size=1,detail_limit=1)
        list_calls=[]
        detail_calls=[]

        def request(url, params):
            if url.endswith('getLostGoodsDetailInfo'):
                detail_calls.append(dict(params))
                return ET.fromstring(
                    '<response><header><resultCode>00</resultCode></header>'
                    '<body><item><atcId>L1</atcId><lstPrdtNm>지갑</lstPrdtNm>'
                    '<lstYmd>2026-03-15</lstYmd><lstPlace>강남역</lstPlace>'
                    '</item></body></response>'
                )
            list_calls.append(dict(params))
            day=params['START_YMD']
            formatted=f'{day[:4]}-{day[4:6]}-{day[6:]}'
            return ET.fromstring(
                '<response><header><resultCode>00</resultCode></header>'
                '<body><totalCount>1</totalCount><items><item>'
                f'<atcId>L{len(list_calls)}</atcId><lstPrdtNm>지갑</lstPrdtNm>'
                f'<lstYmd>{formatted}</lstYmd>'
                '</item></items></body></response>'
            )

        with patch('jupjup.api_client.API_DEFINITIONS',(API_DEFINITIONS[0],)), patch.object(client,'_request_xml',side_effect=request):
            responses, errors=client.search_all(LostItemQuery(item_name='지갑',lost_date=date(2026,3,14)))

        self.assertEqual(len(list_calls),3)
        self.assertEqual(len(detail_calls),1)
        self.assertFalse(errors)
        self.assertEqual(responses[0].records[0].event_place,'강남역')

    def test_expired_budget_does_not_start_request(self):
        client=Lost112ApiClient('test',detail_limit=0)
        with patch.object(client,'_request_xml') as request:
            result=client._search_window(API_DEFINITIONS[1],LostItemQuery(item_name='지갑'),date(2026,3,14),date(2026,3,20),0)
        request.assert_not_called()
        self.assertFalse(result.search_scopes[0].complete)
        self.assertIn('TimeoutError',result.search_scopes[0].error)

    def test_dated_search_uses_three_nonoverlapping_periods(self):
        client = Lost112ApiClient('test', detail_limit=0)
        calls = []
        def request(url, params):
            calls.append((url, dict(params)))
            return xml([])
        with patch.object(client, '_request_xml', side_effect=request):
            client.search_all(LostItemQuery(item_name='지갑', lost_date=date(2026, 3, 14)))
        found = [(url, p) for url, p in calls if '/LosfundInfo' in url]
        self.assertEqual([(p.get('START_YMD'), p.get('END_YMD')) for _, p in found], [('20260314','20260320'), ('20260321','20260327'), ('20260328','20260427')])
        self.assertTrue(all(url.endswith('getLosfundInfoAccToClAreaPd') for url, _ in found))
        self.assertTrue(all('PRDT_NM' not in p for _, p in found))

    def test_pagination_filters_other_items_and_out_of_range_dates(self):
        client = Lost112ApiClient('test', page_size=1, detail_limit=0)
        calls = []
        def request(url, params):
            calls.append(dict(params))
            if params.get('START_YMD') != '20260314':
                return xml([])
            pages = {1: [('card','신용카드','2026-03-15')], 2:[('wallet','샤넬 지갑','2026-03-16')], 3:[('recent','지갑','2026-09-10')]}
            return xml(pages[int(params['pageNo'])], 3)
        with patch('jupjup.api_client.API_DEFINITIONS', (API_DEFINITIONS[1],)), patch.object(client, '_request_xml', side_effect=request):
            responses, errors = client.search_all(LostItemQuery(item_name='지갑', lost_date=date(2026,3,14)))
        self.assertEqual([r.atc_id for response in responses for r in response.records], ['wallet'])
        self.assertEqual([p['pageNo'] for p in calls[:3]], ['1','2','3'])
        self.assertFalse(errors)
        self.assertEqual(
            responses[0].search_scopes[0].partial_reason,
            '요청 기간 밖 또는 날짜 미상 자료 제외',
        )

    def test_five_candidates_stop_expansion_and_preserve_scope(self):
        client = Lost112ApiClient('test', detail_limit=0)
        calls=[]
        def request(url, params):
            calls.append(params)
            return xml([(str(i),'지갑','2026-03-15') for i in range(5)])
        with patch('jupjup.api_client.API_DEFINITIONS', (API_DEFINITIONS[1],)), patch.object(client,'_request_xml',side_effect=request):
            result=JupJupAgentService(client).run(LostItemQuery(item_name='지갑',lost_date=date(2026,3,14)))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(result.candidates),5)
        self.assertTrue(getattr(result,'search_scopes',[]))

    def test_second_page_failure_keeps_first_page_records(self):
        client=Lost112ApiClient('test',page_size=1,detail_limit=0)
        def request(url, params):
            if params.get('START_YMD') != '20260314': return xml([])
            if params['pageNo']=='2': raise TimeoutError('test timeout')
            return xml([('wallet','지갑','2026-03-15')],2)
        with patch('jupjup.api_client.API_DEFINITIONS', (API_DEFINITIONS[1],)), patch.object(client,'_request_xml',side_effect=request):
            responses, errors=client.search_all(LostItemQuery(item_name='지갑',lost_date=date(2026,3,14)))
        self.assertEqual([r.atc_id for response in responses for r in response.records],['wallet'])
        self.assertTrue(errors)
        with patch('jupjup.api_client.API_DEFINITIONS', (API_DEFINITIONS[1],)), patch.object(
            client, '_request_xml', side_effect=request
        ):
            result = JupJupAgentService(client).run(
                LostItemQuery(item_name='지갑', lost_date=date(2026,3,14))
            )
        self.assertEqual(result.source_counts, {'경찰청 습득물': 2})

    def test_dated_search_distinguishes_zero_results_from_total_failure(self):
        client=Lost112ApiClient('test',detail_limit=0)
        definition=(API_DEFINITIONS[1],)

        with patch('jupjup.api_client.API_DEFINITIONS',definition), patch.object(
            client,'_request_xml',return_value=xml([])
        ):
            zero_result=JupJupAgentService(client).run(
                LostItemQuery(item_name='지갑',lost_date=date(2026,3,14))
            )

        self.assertEqual(zero_result.source_counts, {'경찰청 습득물': 0})
        self.assertFalse(zero_result.errors)
        self.assertTrue(zero_result.search_scopes[0].complete)

        for failure in (
            urllib.error.URLError('upstream down'),
            TimeoutError('request timeout'),
        ):
            with self.subTest(failure=type(failure).__name__), patch(
                'jupjup.api_client.API_DEFINITIONS',definition
            ), patch.object(client,'_request_xml',side_effect=failure):
                failed=JupJupAgentService(client).run(
                    LostItemQuery(item_name='지갑',lost_date=date(2026,3,14))
                )

            self.assertEqual(failed.source_counts, {})
            self.assertIn('경찰청 습득물', failed.errors)
            self.assertTrue(failed.search_scopes)
            self.assertTrue(all(scope.pages_completed == 0 for scope in failed.search_scopes))

    def test_future_date_does_not_search_latest(self):
        client=Lost112ApiClient('test',detail_limit=0)
        with patch.object(client,'_request_xml',return_value=xml([])) as request:
            with self.assertRaises(ValueError):
                client.search_all(LostItemQuery(item_name='지갑',lost_date=date(9998,1,1)))
        request.assert_not_called()


if __name__ == '__main__': unittest.main()
