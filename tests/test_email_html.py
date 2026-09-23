import unittest
from services.email_html import sanitize_email_html
from services import admin_api
from unittest.mock import patch
from types import SimpleNamespace

class EmailHTMLTests(unittest.TestCase):
    def test_preserves_inline_design_and_removes_active_content(self):
        result = sanitize_email_html('<div style="padding:24px;background-color:#fff;background:url(https://bad);position:fixed" onclick="bad()"><h1>Hello</h1><script>bad()</script><img src="https://tracker"><a href="javascript:bad()">Bad</a><a href="https://example.com">Try it</a></div>')
        self.assertIn('padding:24px;background-color:#fff', result)
        self.assertIn('href="https://example.com"', result)
        for unsafe in ('script','onclick','javascript:','position:','url(','<img'):
            self.assertNotIn(unsafe, result)
    def test_layout_uses_design_and_keeps_footer(self):
        sender=SimpleNamespace(_from_address=lambda:'Team',_footer=lambda uid:'FOOTER')
        with patch('services.email_sender',sender,create=True):
            result=admin_api.layout_preview(admin_api.LayoutPreview(html='<h1 style="color:#e88d7d">News</h1>'))
        self.assertIn('color:#e88d7d',result['html'])
        self.assertTrue(result['html'].endswith('FOOTER'))
        self.assertNotIn('Your message will appear here',result['html'])
