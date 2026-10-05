import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI, HTTPException
import asyncio
import httpx
from services import admin_api as api

class AdminTests(unittest.TestCase):
    def test_ai_draft_only_generates_copy(self):
        with patch('services.admin_email_ai.draft_email',new_callable=AsyncMock,return_value={'subject':'News','message':'Hello'}) as generate:
            result=asyncio.run(api.ai_draft(api.AIDraft(instructions='Announce a feature', audience='pro')))
        self.assertEqual(result['subject'],'News')
        generate.assert_awaited_once_with('Announce a feature','','','pro','',False)
        self.db.collection.assert_not_called()
    def test_ai_failure_returns_safe_error(self):
        with patch('services.admin_email_ai.draft_email',new_callable=AsyncMock,side_effect=ValueError('private provider details')):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(api.ai_draft(api.AIDraft(instructions='Write an email')))
        self.assertEqual(error.exception.status_code,502)
        self.assertNotIn('private',error.exception.detail)
    def setUp(self):
        api._student_directory_cache = None
        api._student_directory_expires = 0
        self.auth = SimpleNamespace(verify_id_token=Mock(return_value={'uid':'u', 'email':'admin@example.com', 'email_verified':True}))
        self.modules = patch.dict(sys.modules, {'firebase_admin':SimpleNamespace(auth=self.auth)})
        self.modules.start()
        self.ref = Mock()
        self.ref.get.return_value = SimpleNamespace(exists=True, to_dict=lambda:{'role':'admin','active':True})
        self.db = Mock()
        self.db.collection.return_value.document.return_value = self.ref
        self.dbpatch = patch.object(api, 'database', return_value=self.db)
        self.dbpatch.start()
    def tearDown(self):
        self.modules.stop(); self.dbpatch.stop()
    def request(self, token='Bearer token'):
        return SimpleNamespace(headers={'authorization':token})
    def test_requires_token(self):
        with self.assertRaises(HTTPException) as cm: api.require_admin(self.request(''))
        self.assertEqual(cm.exception.status_code,401)
    def test_rejects_invalid_token(self):
        self.auth.verify_id_token.side_effect=ValueError('invalid')
        with self.assertRaises(HTTPException): api.require_admin(self.request())
    def test_denies_revoked_role(self):
        self.ref.get.return_value.to_dict=lambda:{'role':'admin','active':False}
        with self.assertRaises(HTTPException) as cm: api.require_admin(self.request())
        self.assertEqual(cm.exception.status_code,403)
    def test_regular_admin_cannot_manage_roles(self):
        identity=api.require_admin(self.request())
        self.auth.verify_id_token.assert_called_with('token',check_revoked=True)
        with self.assertRaises(HTTPException): api.require_owner(identity)
    def test_owner_is_verified_and_bootstrapped(self):
        self.auth.verify_id_token.return_value={'uid':'owner','email':api.OWNER_EMAIL,'email_verified':True}
        self.ref.get.return_value.exists=False
        self.assertEqual(api.require_admin(self.request())['role'],'owner')
        self.ref.set.assert_called_once()
    def test_unverified_owner_is_denied(self):
        self.auth.verify_id_token.return_value={'uid':'owner','email':api.OWNER_EMAIL,'email_verified':False}
        with self.assertRaises(HTTPException): api.require_admin(self.request())
        self.ref.set.assert_not_called()
    def test_owner_cannot_be_revoked(self):
        with self.assertRaises(HTTPException): api.revoke('owner',{'uid':'owner','role':'owner'})
    def test_preview_uses_auth_email_and_does_not_send(self):
        self.auth.get_user=Mock(return_value=SimpleNamespace(uid='u',email='real@example.com',disabled=False))
        sender=SimpleNamespace(is_suppressed=Mock(return_value=None),is_enabled=lambda:False,send_email=Mock(),_from_address=lambda:'Team',_footer=lambda uid:'<footer>Unsubscribe</footer>')
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            result=api.preview(api.Draft(uid='u',subject='Hello',message='How did your exam go?'),{'uid':'admin'})
        self.assertEqual(result['to'],'real@example.com')
        self.assertFalse(result['sendingEnabled'])
        self.assertIn('<footer>Unsubscribe</footer>', result['html'])
        sender.send_email.assert_not_called()
    def test_email_lookup_resolves_existing_account(self):
        self.auth.get_user_by_email=Mock(return_value=SimpleNamespace(uid='recipient',email='real@example.com',disabled=False))
        sender=SimpleNamespace(is_suppressed=Mock(return_value=None),is_enabled=lambda:False,_from_address=lambda:'Team',_footer=lambda uid:'footer')
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            result=api.preview(api.Draft(email=' REAL@example.com ',subject='Hi',message='Hello'),{'uid':'admin'})
        self.auth.get_user_by_email.assert_called_once_with('real@example.com')
        self.assertEqual(result['uid'],'recipient')
        sender.is_suppressed.assert_called_once_with(self.db,'recipient')
    def test_template_escapes_html_and_preserves_paragraphs(self):
        rendered=api.personal_email_html('<script>alert(1)</script>\n\nHi\nthere')
        self.assertNotIn('<script>',rendered)
        self.assertIn('&lt;script&gt;',rendered)
        self.assertIn('Hi<br>there',rendered)
    def test_campaign_deduplicates_and_excludes_opted_out_users_without_sending(self):
        self.auth.get_user_by_email=Mock(side_effect=lambda email:SimpleNamespace(uid=email,email=email,disabled=False))
        sender=SimpleNamespace(is_suppressed=lambda db,uid:'unsubscribed' if uid=='out@example.com' else None,
            is_enabled=lambda:False,_from_address=lambda:'Team',_footer=lambda uid:'footer',remaining_today=lambda db:20,send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            result=api.campaign_preview(api.CampaignDraft(audience='selected',emails=['A@example.com','a@example.com','out@example.com'],subject='News',message='New feature'),{'uid':'admin'})
        self.assertEqual(result['total'],1)
        self.assertEqual(result['excluded'],1)
        self.assertEqual(result['recipients'][0]['to'],'a@example.com')
        sender.send_email.assert_not_called()
    def test_campaign_cap_preserves_pending_recipients(self):
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        from datetime import datetime,timezone
        data={'kind':'campaign','actor':'admin','status':'draft','sendingEnabled':False,'createdAt':datetime.now(timezone.utc),
              'recipients':[{'uid':'u','to':'u@example.com','status':'pending'}]}
        self.ref.get.return_value.to_dict=lambda:data
        sender=SimpleNamespace(is_enabled=lambda:False,remaining_today=lambda db:0,send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            result=api.campaign_send('id',{'uid':'admin'})
        self.assertEqual(result['status'],'paused')
        self.assertEqual(result['counts']['pending'],1)
        self.assertFalse(result['canContinue'])
        sender.send_email.assert_not_called()
    def test_campaign_resumes_only_pending_recipients(self):
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        self.auth.get_user=Mock(return_value=SimpleNamespace(email='new@example.com',disabled=False))
        data={'kind':'campaign','actor':'admin','status':'paused','sendingEnabled':False,'subject':'News','message':'Hi',
              'recipients':[{'uid':'old','to':'old@example.com','status':'sent'},{'uid':'new','to':'new@example.com','status':'pending'}]}
        self.ref.get.return_value.to_dict=lambda:data
        sender=SimpleNamespace(is_enabled=lambda:False,remaining_today=lambda db:20,send_email=Mock(return_value={'status':'dry_run'}))
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            result=api.campaign_send('id',{'uid':'admin'})
        self.assertEqual(result['status'],'completed')
        sender.send_email.assert_called_once()
        self.assertEqual(sender.send_email.call_args.kwargs['uid'],'new')
    def test_send_rejects_duplicate_and_other_admin_drafts(self):
        firebase=sys.modules['firebase_admin']
        firebase.firestore=SimpleNamespace(transactional=lambda fn:fn)
        sender=SimpleNamespace(send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            for data in ({'actor':'someone-else','status':'draft'}, {'actor':'admin','status':'sending'}, {'actor':'admin','status':'sent'}):
                self.ref.get.return_value.to_dict=lambda:data
                with self.assertRaises(HTTPException): api.send('id',{'uid':'admin'})
        sender.send_email.assert_not_called()
    def test_single_send_rejects_mode_change_before_reserving(self):
        from datetime import datetime, timezone
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        self.ref.get.return_value.to_dict=lambda:{'actor':'admin','status':'draft','sendingEnabled':False,
            'createdAt':datetime.now(timezone.utc)}
        sender=SimpleNamespace(is_enabled=lambda:True,send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            with self.assertRaises(HTTPException) as error: api.send('id',{'uid':'admin'})
        self.assertIn('mode changed',error.exception.detail)
        self.db.transaction.return_value.update.assert_not_called()
        sender.send_email.assert_not_called()
    def test_single_send_rechecks_account_address(self):
        from datetime import datetime, timezone
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        self.auth.get_user=Mock(return_value=SimpleNamespace(email='changed@example.com',disabled=False))
        self.ref.get.return_value.to_dict=lambda:{'actor':'admin','status':'draft','sendingEnabled':False,
            'uid':'student','to':'old@example.com','createdAt':datetime.now(timezone.utc)}
        sender=SimpleNamespace(is_enabled=lambda:False,send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            with self.assertRaises(HTTPException) as error: api.send('id',{'uid':'admin'})
        self.assertIn('account changed',error.exception.detail)
        self.db.transaction.return_value.update.assert_not_called()
        sender.send_email.assert_not_called()
    def test_campaign_setup_failure_does_not_lock_campaign(self):
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        sender=SimpleNamespace(is_enabled=lambda:True,configuration_issues=lambda:['Set RESEND_API_KEY.'],send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            with self.assertRaises(HTTPException) as error: api.campaign_send('id',{'uid':'admin'})
        self.assertEqual(error.exception.status_code,409)
        self.db.transaction.assert_not_called()
        sender.send_email.assert_not_called()
    def test_paused_campaign_review_does_not_send(self):
        self.ref.get.return_value.to_dict=lambda:{'actor':'admin','kind':'campaign','status':'paused',
            'sendingEnabled':False,'subject':'News','message':'Hello','recipients':[{'uid':'student','status':'pending'}]}
        sender=SimpleNamespace(is_enabled=lambda:False,_from_address=lambda:'Team',remaining_today=lambda db:10,
            _footer=lambda uid:'<footer>'+uid+'</footer>',send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            result=api.saved_email_preview('id',{'uid':'admin'})
        self.assertIn('<footer>student</footer>',result['html'])
        self.assertEqual(result['remainingToday'],10)
        self.ref.update.assert_not_called()
        sender.send_email.assert_not_called()
    def test_every_workspace_route_rejects_anonymous_requests(self):
        app=FastAPI(); app.include_router(api.router)
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                for method,path in [('get','/me'),('get','/users'),('get','/exams'),('get','/email'),('get','/email/settings'),('get','/email/id/preview'),('post','/email/layout-preview'),('post','/email/ai-draft'),('get','/admins'),('post','/admins'),('post','/email/preview'),('post','/email/campaign/preview'),('post','/email/draft/batch'),('post','/email/draft/send'),('delete','/admins/u')]:
                    with self.subTest(path=path):
                        response = await getattr(client, method)('/admin/workspace'+path)
                        self.assertEqual(response.status_code,401)
        asyncio.run(check())

    def test_incomplete_working_draft_is_saved_but_cannot_be_sent(self):
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        result=api.save_working_email(api.WorkingEmail(message='Still writing'),{'uid':'admin'})
        saved=self.ref.create.call_args.args[0]
        self.assertEqual(saved['status'],'composing')
        self.assertEqual(saved['subject'],'')
        self.assertTrue(result['id'])
        self.ref.get.return_value.to_dict=lambda:saved
        with self.assertRaises(HTTPException): api.send(result['id'],{'uid':'admin'})

    def test_working_draft_update_enforces_ownership_and_state(self):
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        for data in ({'actor':'another','status':'composing'},{'actor':'admin','status':'sent'}):
            self.ref.get.return_value.to_dict=lambda:data
            with self.assertRaises(HTTPException): api.save_working_email(api.WorkingEmail(draft_id='id'),{'uid':'admin'})
        self.db.transaction.return_value.update.assert_not_called()

    def test_templates_strip_unsafe_markup_and_have_no_recipients(self):
        api.save_email_template(api.EmailTemplate(name='Hello',message='Hi',html='<p onclick="bad()">Hi</p><script>bad()</script>'),{'uid':'admin'})
        data=self.ref.create.call_args.args[0]
        self.assertEqual(data['html'],'<p>Hi</p>')
        self.assertNotIn('email',data)
        self.assertNotIn('recipients',data)

    def test_student_search_matches_names_and_skips_unsubscribed(self):
        docs=[SimpleNamespace(id='a',to_dict=lambda:{'displayName':'Alex Nurse','email':'alex@example.com'}),
              SimpleNamespace(id='b',to_dict=lambda:{'displayName':'Alex Other','email':'other@example.com','emailPrefs':{'marketing':False}}),
              SimpleNamespace(id='c',to_dict=lambda:{'displayName':'Sam Nurse','email':'sam@example.com'})]
        self.db.collection.return_value.select.return_value.stream.return_value=docs
        result=api.email_students(q='ALEX')
        self.assertEqual([item['uid'] for item in result['items']],['a'])
        self.assertIsNone(result['cursor'])

    def test_sparse_student_search_covers_the_full_directory_and_reuses_cache(self):
        docs=[SimpleNamespace(id=str(i),to_dict=lambda:{'displayName':'Sam','email':'sam@example.com'}) for i in range(501)]
        docs.append(SimpleNamespace(id='target',to_dict=lambda:{'displayName':'Alex','email':'alex@example.com'}))
        stream = self.db.collection.return_value.select.return_value.stream
        stream.return_value=docs
        self.assertEqual(api.email_students(q='alex')['items'][0]['uid'],'target')
        result=api.email_students(q='not matched')
        self.assertEqual(result['items'],[])
        self.assertIsNone(result['cursor'])
        stream.assert_called_once()

    def test_student_search_paginates_matches_and_refreshes_expired_cache(self):
        docs=[SimpleNamespace(id=f'{i:03}',to_dict=lambda:{'name':'Sam','email':'sam@example.com'}) for i in range(35)]
        stream = self.db.collection.return_value.select.return_value.stream
        stream.return_value=docs
        first=api.email_students(q='sam')
        self.assertEqual(len(first['items']),30)
        second=api.email_students(q='sam',cursor=first['cursor'])
        self.assertEqual(len(second['items']),5)
        self.assertIsNone(second['cursor'])
        self.assertFalse({s['uid'] for s in first['items']} & {s['uid'] for s in second['items']})
        api._student_directory_expires=0
        stream.return_value=[]
        self.assertEqual(api.email_students(q='sam')['items'],[])
        self.assertEqual(stream.call_count,2)

    def test_test_send_targets_verified_admin_and_does_not_modify_campaign(self):
        from datetime import datetime,timezone
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        self.auth.get_user=Mock(return_value=SimpleNamespace(uid='admin',email='secretary@example.com',email_verified=True,disabled=False))
        self.ref.get.return_value.to_dict=lambda:{'actor':'admin','status':'draft','sendingEnabled':True,
            'createdAt':datetime.now(timezone.utc),'subject':'News','message':'Hello','html':'<p><b>Hello</b></p>',
            'recipients':[{'uid':'student','to':'student@example.com','status':'pending'}]}
        test_ref=Mock()
        test_ref.get.return_value.to_dict=lambda:{}
        self.db.collection.side_effect=lambda name:SimpleNamespace(document=lambda key:test_ref if name=='adminMailTests' else self.ref,add=Mock())
        sender=SimpleNamespace(is_enabled=lambda:True,configuration_issues=lambda:[],remaining_today=lambda db:50,send_email=Mock(return_value={'status':'sent'}))
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            result=api.send_email_test('review1',{'uid':'admin','email':'secretary@example.com'})
        kwargs=sender.send_email.call_args.kwargs
        self.assertEqual(kwargs['to'],'secretary@example.com')
        self.assertEqual(kwargs['uid'],'admin')
        self.assertEqual(kwargs['subject'],'[Test] News')
        self.assertIn('<p><b>Hello</b></p>',kwargs['html'])
        self.assertIn('font:16px Arial',kwargs['html'])
        self.assertTrue(kwargs['transactional'])
        self.assertEqual(result['status'],'sent')
        self.ref.update.assert_not_called()
        self.db.transaction.return_value.set.assert_called_once()

    def test_test_send_rejects_another_admin_and_practice_mode(self):
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        sender=SimpleNamespace(is_enabled=lambda:False,send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            for actor in ('someone-else','admin'):
                self.ref.get.return_value.to_dict=lambda:{'actor':actor,'status':'draft','sendingEnabled':False}
                with self.assertRaises(HTTPException): api.send_email_test('review1',{'uid':'admin','email':'secretary@example.com'})
        sender.send_email.assert_not_called()

    def test_test_send_does_not_repeat_or_exceed_budget(self):
        from datetime import datetime,timezone
        sys.modules['firebase_admin'].firestore=SimpleNamespace(transactional=lambda fn:fn)
        self.auth.get_user=Mock(return_value=SimpleNamespace(uid='admin',email='secretary@example.com',email_verified=True,disabled=False))
        self.ref.get.return_value.to_dict=lambda:{'actor':'admin','status':'draft','sendingEnabled':True,'createdAt':datetime.now(timezone.utc)}
        test_ref=Mock()
        self.db.collection.side_effect=lambda name:SimpleNamespace(document=lambda key:test_ref if name=='adminMailTests' else self.ref)
        sender=SimpleNamespace(is_enabled=lambda:True,configuration_issues=lambda:[],remaining_today=lambda db:0,send_email=Mock())
        with patch.dict(sys.modules, {'services.email_sender':sender}), patch('services.email_sender',sender,create=True):
            test_ref.get.return_value.to_dict=lambda:{'result':{'status':'sent','to':'secretary@example.com'}}
            self.assertEqual(api.send_email_test('id',{'uid':'admin','email':'secretary@example.com'})['status'],'sent')
            for previous in ({'status':'sending'}, {}):
                test_ref.get.return_value.to_dict=lambda:previous
                with self.assertRaises(HTTPException): api.send_email_test('id',{'uid':'admin','email':'secretary@example.com'})
            self.auth.get_user.return_value.email='changed@example.com'
            with self.assertRaises(HTTPException): api.send_email_test('id',{'uid':'admin','email':'secretary@example.com'})
        sender.send_email.assert_not_called()
        self.db.transaction.return_value.set.assert_not_called()

    def test_new_email_routes_require_authentication(self):
        app=FastAPI(); app.include_router(api.router)
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                for method,path in [('get','/email/students'),('get','/email/templates'),('post','/email/templates'),('post','/email/drafts'),('post','/email/id/test')]:
                    response=await getattr(client,method)('/admin/workspace'+path)
                    self.assertEqual(response.status_code,401)
        asyncio.run(check())

if __name__=='__main__': unittest.main()


