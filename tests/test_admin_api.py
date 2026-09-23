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
    def test_every_workspace_route_rejects_anonymous_requests(self):
        app=FastAPI(); app.include_router(api.router)
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                for method,path in [('get','/me'),('get','/users'),('get','/exams'),('get','/email'),('get','/admins'),('post','/admins'),('post','/email/preview'),('post','/email/campaign/preview'),('post','/email/draft/batch'),('post','/email/draft/send'),('delete','/admins/u')]:
                    with self.subTest(path=path):
                        response = await getattr(client, method)('/admin/workspace'+path)
                        self.assertEqual(response.status_code,401)
        asyncio.run(check())

if __name__=='__main__': unittest.main()


