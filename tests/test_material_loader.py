"""Extraction must retain teaching structure and late visual-only pages."""
import tempfile
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from core.material_loader import TeachingMaterialLoader


class LoaderTests(unittest.TestCase):
    def test_word_preserves_options_breaks_tables_emphasis_and_footer(self):
        from docx import Document
        from docx.shared import Inches
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root/'figure.png'
            Image.new('RGB', (32,32), 'white').save(image)
            document = Document()
            document.add_heading('Learning objectives', 1)
            paragraph = document.add_paragraph('Classify health models.')
            paragraph.add_run(' Must know!').bold = True
            paragraph.add_run(' Ordinary text.').bold = False
            question = document.add_paragraph('Which model applies?')
            question.add_run().add_break()
            question.add_run('A) Clinical model')
            table = document.add_table(rows=1, cols=2)
            table.cell(0,0).text = 'Clinical model'
            table.cell(0,1).text = 'Absence of disease'
            table.cell(0,1).paragraphs[0].add_run().add_picture(str(image), width=Inches(.2))
            document.sections[0].footer.paragraphs[0].text = 'Family theories will not be tested.'
            path = root/'notes.docx'; document.save(path)
            with patch('core.material_loader.describe_visual', return_value='[Table figure]\nDiagram labels: Clinical model') as read:
                text = '\n'.join(TeachingMaterialLoader(str(path))._word())
            self.assertIn('[Heading: Learning objectives]', text)
            self.assertIn('Which model applies?\nA) Clinical model', text)
            self.assertIn('Clinical model | Absence of disease', text)
            self.assertIn('[Author emphasis:  Must know!]', text)
            self.assertNotIn('[Author emphasis:  Ordinary text.', text)
            self.assertIn('Family theories will not be tested.', text)
            self.assertIn('Diagram labels: Clinical model', text)
            read.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF is installed in the app environment, not this bundled test runtime')
    def test_pdf_reads_an_image_only_page_after_text_pages(self):
        import fitz
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); image = root/'figure.png'
            Image.new('RGB',(80,80),'white').save(image)
            document = fitz.open()
            for index in range(4):
                page = document.new_page()
                page.insert_text((40,40), f'Learning objective {index}. ' + 'Recognize the taught health framework. '*5)
            page = document.new_page(); page.insert_image(fitz.Rect(20,20,100,100),filename=str(image))
            path = root/'mixed.pdf'; document.save(path); document.close()
            with patch('core.material_loader.describe_visual',return_value='Final page: MUST KNOW family assessment') as read:
                pages = TeachingMaterialLoader(str(path))._pdf()
            self.assertEqual(len(pages),5)
            self.assertIn('Final page: MUST KNOW family assessment',pages[-1])
            self.assertEqual(read.call_args.args[-1], 'PDF page 5')

    def test_slides_keep_speaker_notes_and_tables(self):
        from pptx import Presentation
        from pptx.util import Inches
        with tempfile.TemporaryDirectory() as folder:
            deck = Presentation(); slide = deck.slides.add_slide(deck.slide_layouts[6])
            slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1)).text = 'Learning objective: use SBAR'
            table = slide.shapes.add_table(1,2,Inches(1),Inches(2), Inches(5), Inches(1)).table
            table.cell(0,0).text = 'S'; table.cell(0,1).text = 'Situation'
            slide.notes_slide.notes_text_frame.text = 'Exam instruction: recognize each SBAR component.'
            path = Path(folder)/'notes.pptx'; deck.save(path)
            text = '\n'.join(TeachingMaterialLoader(str(path))._slides())
            self.assertIn('Learning objective: use SBAR',text)
            self.assertIn('S | Situation',text)
            self.assertIn('Exam instruction: recognize each SBAR component.',text)


if __name__ == '__main__': unittest.main()
