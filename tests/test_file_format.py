"""Uploads are routed by content, and a file problem says what to do about it."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from core import file_format
from core.file_format import UploadFormatError, convert_legacy, detect_format
from core.material_loader import TeachingMaterialLoader


def deck(path, *texts, picture=None):
    from pptx import Presentation
    from pptx.util import Inches
    presentation = Presentation()
    for text in texts:
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = text
        if picture:
            slide.shapes.add_picture(str(picture), Inches(1), Inches(2), Inches(1))
    presentation.save(path)
    return path


def word(path, text):
    from docx import Document
    document = Document()
    document.add_paragraph(text)
    document.save(path)
    return path


class DetectFormatTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return str(path)

    def code(self, path):
        with self.assertRaises(UploadFormatError) as raised:
            detect_format(path)
        return raised.exception.code

    def test_the_content_wins_over_the_name(self):
        self.assertEqual(detect_format(self.write('notes.pdf', b'%PDF-1.7\n...')), '.pdf')
        self.assertEqual(detect_format(str(deck(self.root / 'week3.pdf', 'Shock'))), '.pptx')
        self.assertEqual(detect_format(str(word(self.root / 'week3.pptx', 'Shock'))), '.docx')
        self.assertEqual(detect_format(self.write('scan.jpg', b'\x89PNG\r\n\x1a\n' + b'\x00' * 20)), '.png')

    def test_a_pdf_header_after_leading_bytes_is_still_a_pdf(self):
        self.assertEqual(detect_format(self.write('notes.pdf', b'\xef\xbb\xbf\r\n%PDF-1.4\n')), '.pdf')

    def test_legacy_office_binaries_are_recognised_for_conversion(self):
        ole = file_format.OLE_MAGIC + b'\x00' * 100
        self.assertEqual(detect_format(self.write('lecture.ppt', ole)), '.ppt')
        self.assertEqual(detect_format(self.write('notes.doc', ole)), '.doc')
        # Saved by old PowerPoint but renamed .pptx by hand.
        self.assertEqual(detect_format(self.write('lecture.pptx', ole)), '.doc')

    def test_markdown_and_unknown_text_are_read_as_text(self):
        self.assertEqual(detect_format(self.write('notes.md', '# Shock\nSigns: hypotension'.encode())), '.txt')
        self.assertEqual(detect_format(self.write('notes', 'Préparation à l’examen'.encode())), '.txt')
        self.assertEqual(detect_format(self.write('labs.csv', b'test,value\nK,5.9\n')), '.csv')

    def test_a_renamed_or_damaged_office_file_is_named_as_such(self):
        self.assertEqual(self.code(self.write('lecture.pptx', b'<html><body>Sign in</body></html>')),
                         file_format.NOT_A_DOCUMENT)

    def test_unreadable_content_is_unsupported_not_retried(self):
        self.assertEqual(self.code(self.write('clip.mp4', b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 50)),
                         file_format.UNSUPPORTED_TYPE)
        archive = self.root / 'unit.zip'
        with zipfile.ZipFile(archive, 'w') as handle:
            handle.writestr('slides/one.txt', 'x')
        self.assertEqual(self.code(str(archive)), file_format.UNSUPPORTED_TYPE)

    def test_an_empty_file_says_so(self):
        self.assertEqual(self.code(self.write('empty.pdf', b'')), file_format.UNREADABLE_DOCUMENT)

    def test_without_libreoffice_a_legacy_file_gets_the_instruction(self):
        path = self.write('lecture.ppt', file_format.OLE_MAGIC + b'\x00' * 100)
        with patch('core.file_format.shutil.which', return_value=None):
            with self.assertRaises(UploadFormatError) as raised:
                convert_legacy(path, '.ppt', self.folder.name)
        self.assertEqual(raised.exception.code, file_format.LEGACY_OFFICE)
        self.assertIn('.pptx', str(raised.exception))


class SlideRobustnessTests(unittest.TestCase):
    def test_a_linked_picture_costs_one_figure_not_the_deck(self):
        from PIL import Image
        from pptx.shapes.picture import Picture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'figure.png'
            Image.new('RGB', (32, 32), 'white').save(image)
            path = deck(root / 'shock.pptx', 'Types of shock', 'Nursing priorities', picture=image)

            def linked(self):
                raise ValueError('no embedded image')
            with patch.object(Picture, 'image', property(linked)):
                documents = TeachingMaterialLoader(str(path)).load()
        text = '\n'.join(d.page_content for d in documents)
        self.assertIn('Types of shock', text)
        self.assertIn('Nursing priorities', text)
        self.assertIn('[UNREADABLE FIGURE', text)

    def test_a_package_python_pptx_refuses_is_read_from_its_xml(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good = deck(root / 'good.pptx', 'Hyperkalemia', 'Peaked T waves')
            broken = root / 'broken.pptx'
            # Drop the package relationships: python-pptx then can't find the
            # presentation part, the same refusal seen in production.
            with zipfile.ZipFile(good) as source, zipfile.ZipFile(broken, 'w') as target:
                for item in source.infolist():
                    if item.filename != '_rels/.rels':
                        target.writestr(item, source.read(item.filename))
            documents = TeachingMaterialLoader(str(broken), kind='.pptx').load()
        text = '\n'.join(d.page_content for d in documents)
        self.assertIn('Hyperkalemia', text)
        self.assertIn('Peaked T waves', text)
        self.assertLess(text.index('[Slide 1]'), text.index('[Slide 2]'))

    def test_a_package_with_no_text_at_all_asks_for_a_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / 'broken.pptx'
            with zipfile.ZipFile(broken, 'w') as target:
                target.writestr('ppt/presentation.xml', '<p/>')
            with self.assertRaises(UploadFormatError) as raised:
                TeachingMaterialLoader(str(broken), kind='.pptx').load()
        self.assertEqual(raised.exception.code, file_format.UNREADABLE_DOCUMENT)

    def test_a_deck_named_pdf_is_read_as_a_deck(self):
        with tempfile.TemporaryDirectory() as folder:
            path = deck(Path(folder) / 'week3.pdf', 'Fluid balance')
            documents = TeachingMaterialLoader(str(path), kind='.pptx').load()
        self.assertIn('Fluid balance', '\n'.join(d.page_content for d in documents))


if __name__ == '__main__':
    unittest.main()
