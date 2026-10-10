"""Exercise membership routing and real SDK serialization without API calls."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.material_model import MODEL_POLICY, service_tier_for_chat
from core.material_loader import TeachingMaterialLoader, describe_visual
from services.document_understanding import MaterialError, analyse_document, model_json, fingerprint
from services.material_practice import stream_material_practice, plan_key


class MembershipTests(unittest.TestCase):
    def test_membership_is_refreshed_and_only_pro_receives_fast(self):
        data = {'usage': {'tier': 'free'}}
        snapshot = SimpleNamespace(to_dict=lambda: data)
        db = SimpleNamespace(collection=lambda name: SimpleNamespace(document=lambda uid:
                             SimpleNamespace(get=lambda: snapshot)))
        with patch('services.usage_guard._resolve_uid', return_value='owner') as owner, \
             patch('firebase_admin.firestore.client', return_value=db):
            self.assertEqual(service_tier_for_chat('chat'), 'default')
            data['usage']['tier'] = 'pro'
            self.assertEqual(service_tier_for_chat('chat'), 'fast')
            data['usage']['tier'] = 'free'
            self.assertEqual(service_tier_for_chat('chat'), 'default')
            data.clear()
            self.assertEqual(service_tier_for_chat('chat'), 'default')
            self.assertEqual(owner.call_count, 4)

    def test_lookup_failure_and_missing_chat_use_standard(self):
        self.assertEqual(service_tier_for_chat(None), 'default')
        with patch('services.usage_guard._resolve_uid', side_effect=RuntimeError('offline')):
            self.assertEqual(service_tier_for_chat('chat'), 'default')


class RequestTests(unittest.TestCase):
    def setUp(self):
        import httpx
        from langchain_openai import ChatOpenAI
        self.requests = []
        self.finish_reason = 'stop'
        self.content = '{"relevant":true,"description":"Labels A and B","uncertain":false}'
        def respond(request):
            body = json.loads(request.content)
            self.requests.append(body)
            return httpx.Response(200, json={
                'id':'chatcmpl-test', 'object':'chat.completion', 'created':1,
                'model':'gpt-6-luna', 'service_tier':'default',
                'choices':[{'index':0,'finish_reason':self.finish_reason,
                    'message':{'role':'assistant','content':self.content}}],
                'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130,
                         'completion_tokens_details':{'reasoning_tokens':10}}})
        transport = httpx.MockTransport(respond)
        sync = httpx.Client(transport=transport)
        async_client = httpx.AsyncClient(transport=transport)
        def create(**kwargs):
            return ChatOpenAI(api_key='test-only', http_client=sync,
                              http_async_client=async_client, **kwargs)
        self.client_patch = patch('langchain_openai.ChatOpenAI', side_effect=create)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.addCleanup(sync.close)
        self.addCleanup(lambda: asyncio.run(async_client.aclose()))

    def test_sdk_transmits_explicit_tier_reasoning_and_correct_token_limit(self):
        for tier in ('default', 'fast'):
            for effort in ('low', 'medium'):
                with self.subTest(tier=tier, effort=effort):
                    asyncio.run(model_json('Return JSON.', {'material':'Teaching example'},
                                           service_tier=tier, reasoning_effort=effort))
                    body = self.requests[-1]
                    self.assertEqual(body['model'], 'gpt-6-luna')
                    self.assertEqual(body['service_tier'], tier)
                    self.assertEqual(body['reasoning_effort'], effort)
                    self.assertEqual(body['max_completion_tokens'], 8192 if effort == 'low' else 16384)
                    self.assertEqual(body['response_format'], {'type':'json_object'})
                    self.assertNotIn('temperature', body)
                    self.assertNotIn('max_tokens', body)

    def test_visual_input_uses_pro_tier_and_visual_reasoning_policy(self):
        loader = TeachingMaterialLoader('test.docx', service_tier='fast')
        text = loader._describe_visual(b'test-image', 'image/png', 'Figure')
        self.assertIn('Labels A and B', text)
        body = self.requests[-1]
        self.assertEqual(body['service_tier'], 'fast')
        self.assertEqual(body['reasoning_effort'], MODEL_POLICY['visualReasoning'])
        self.assertEqual(body['messages'][1]['content'][1]['type'], 'image_url')

    def test_numbered_passages_send_every_source_character_only_once(self):
        source = 'E\u0301valuation\u00a0ABCDE\n\nPremière étape.\n'
        passages = [{'id':'p0_1', 'text':source[:21]}, {'id':'p0_2', 'text':source[21:]}]
        asyncio.run(model_json('Return JSON.', {'material':source, 'sourcePassages':passages,
                                               'section':1, 'sectionCount':2}))
        payload = json.loads(self.requests[-1]['messages'][1]['content'])
        self.assertNotIn('material', payload)
        self.assertEqual(''.join(p['text'] for p in payload['sourcePassages']),source)
        self.assertEqual(payload['sectionCount'],2)

    def test_truncated_json_and_visuals_are_rejected(self):
        self.finish_reason = 'length'
        with self.assertRaises(MaterialError):
            asyncio.run(model_json('Return JSON.', {}))
        self.assertIn('UNREADABLE FIGURE', describe_visual(b'image', 'image/png', 'Figure'))

    def test_logs_capture_actual_standard_fallback_and_billable_reasoning(self):
        with self.assertLogs('uvicorn.error.material_model', level='INFO') as logs:
            asyncio.run(model_json('Return JSON.', {}, service_tier='fast'))
        text = '\n'.join(logs.output)
        self.assertIn('requested_tier=fast actual_tier=default', text)
        self.assertIn('reasoning_tokens=10', text)


class WorkflowRoutingTests(unittest.TestCase):
    def test_upload_analysis_passes_server_tier_to_each_section(self):
        async def answer(system, payload, **kwargs):
            return {'goals':[], 'signals':[], 'examples':[], 'warnings':[]}
        with patch('services.document_understanding.load_analysis', return_value=None), \
             patch('services.document_understanding.save_analysis', return_value=True), \
             patch('services.document_understanding.service_tier_for_chat', return_value='fast'), \
             patch('services.document_understanding.model_json', new=AsyncMock(side_effect=answer)) as call:
            # Pass the patched default explicitly because Python captures defaults.
            asyncio.run(analyse_document('Teaching material. ' * 1000, 'test.txt', chat_id='chat', call=call))
            self.assertGreater(call.await_count, 1)
            for invocation in call.await_args_list:
                self.assertEqual(invocation.kwargs['service_tier'], 'fast')
                self.assertEqual(invocation.kwargs['reasoning_effort'], MODEL_POLICY['analysisReasoning'])

    def test_plan_and_question_stages_route_by_membership_not_client_settings(self):
        analysis = {'filename':'notes', 'fingerprint':'abc', 'goals':[
            {'id':'goal','topic':'Topic','outcome':'Outcome','evidence':{'filename':'notes','quote':'Supporting teaching statement'}}],
            'examples':[], 'signals':[], 'complete':True}
        plan = {'id':'plan','total':1,'slots':[{'goalId':'goal','index':0}],
                'summary':'Plan','coverage':{'Topic':1},'formats':{'mcq':1},'uncertainties':[]}
        async def generate(*args, **kwargs):
            self.assertEqual(kwargs['call'].keywords, {'service_tier':tier, 'reasoning_effort':'medium'})
            self.assertEqual(kwargs['draft_call'].keywords, {'service_tier':tier, 'reasoning_effort':'low'})
            return {'question':'Supported question'}
        async def collect(session):
            return [event async for event in stream_material_practice(session=session, topic='Topic',
                difficulty='medium', num_questions=1, question_types=['mcq'])]
        for tier in ('default', 'fast'):
            session = SimpleNamespace(chat_id='chat',user_language='en',
                        practice_profile={'service_tier':'fast','tier':'pro'})
            with patch('services.material_practice.service_tier_for_chat', return_value=tier), \
                 patch('services.material_practice.understand_session', new=AsyncMock(return_value=[analysis])), \
                 patch('services.material_practice.prepare_plan', new=AsyncMock(return_value=plan)) as prepare, \
                 patch('services.material_practice.generate_planned_question', side_effect=generate):
                events = asyncio.run(collect(session))
                self.assertEqual(events[-1]['status'], 'quiz_complete')
                self.assertEqual(prepare.call_args.kwargs['call'].keywords,
                                 {'service_tier':tier,'reasoning_effort':'medium'})

    def test_cache_identity_tracks_model_policy_without_membership(self):
        before = fingerprint('Source notes')
        with patch.dict('core.material_model.MODEL_POLICY', {'analysisReasoning':'high'}):
            self.assertNotEqual(before, fingerprint('Source notes'))
        settings = {'scope':'Topic'}
        before = plan_key([], settings)
        with patch.dict('core.material_model.MODEL_POLICY', {'draftReasoning':'medium'}):
            self.assertNotEqual(before, plan_key([], settings))


if __name__ == '__main__':
    unittest.main()
