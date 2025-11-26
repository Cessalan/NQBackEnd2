import os
import fitz  # PyMuPDF
from google.cloud import vision
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List


def is_scanned_pdf(pdf_path: str, threshold: int = 50) -> bool:
    """
    Detect if a PDF is scanned (image-based) or text-based.
    
    Args:
        pdf_path: Path to PDF file
        threshold: Minimum characters per page to consider "text-based"
        
    Returns:
        True if scanned (needs OCR), False if text-based
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Check first 3 pages (or all if less than 3)
        pages_to_check = min(3, len(doc))
        total_text_length = 0
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text()
            total_text_length += len(text.strip())
        
        doc.close()
        
        # If average text per page < threshold, it's likely scanned
        avg_text_per_page = total_text_length / pages_to_check
        is_scanned = avg_text_per_page < threshold
        
        print(f"📊 PDF Analysis: {avg_text_per_page:.0f} chars/page → {'SCANNED' if is_scanned else 'TEXT-BASED'}")
        
        return is_scanned
        
    except Exception as e:
        print(f"⚠️ Error detecting PDF type: {e}")
        return False  # Assume text-based if detection fails


class OCRPDFLoader(BaseLoader):
    """
    Load scanned PDF files using Google Vision OCR.
    Matches the pattern of OCRImageLoader but for multi-page PDFs.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        print("inside the OCR PDF loader")

    def load(self) -> List[Document]:
        """Extract text from scanned PDF using Google Vision API"""
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("GOOGLE_API_KEY environment variable not set")
        
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        print("Google Vision client loaded")
        
        try:
            # Verify file exists and has content
            if not os.path.exists(self.file_path):
                raise Exception(f"File not found: {self.file_path}")
            
            file_size = os.path.getsize(self.file_path)
            if file_size == 0:
                raise Exception("PDF file is empty")
            
            print(f"Processing PDF: {self.file_path} (size: {file_size} bytes)")
            
            # Open PDF
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            print(f"PDF has {total_pages} pages")
            
            all_text = []
            all_confidence_scores = []
            
            # Process each page
            for page_num in range(total_pages):
                print(f"🔍 OCR processing page {page_num + 1}/{total_pages}...")
                
                page = doc[page_num]
                
                # Convert page to image (2x scale for better OCR quality)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                
                if len(img_bytes) == 0:
                    print(f"⚠️ Page {page_num + 1} is empty, skipping...")
                    continue
                
                print(f"  Read {len(img_bytes)} bytes from page {page_num + 1}")
                
                # Create Vision API image object
                image = vision.Image(content=img_bytes)
                
                # Use document_text_detection for handwriting (same as OCRImageLoader)
                response = client.document_text_detection(image=image)
                
                if response.error.message:
                    raise Exception(f"Vision API Error on page {page_num + 1}: {response.error.message}")
                
                # Extract text
                page_text = ""
                if response.full_text_annotation:
                    page_text = response.full_text_annotation.text
                
                # Calculate confidence for this page
                if response.full_text_annotation and response.full_text_annotation.pages:
                    for page_data in response.full_text_annotation.pages:
                        for block in page_data.blocks:
                            if hasattr(block, 'confidence'):
                                all_confidence_scores.append(block.confidence)
                
                if page_text:
                    all_text.append(page_text)
                    print(f"  ✅ Extracted {len(page_text)} characters from page {page_num + 1}")
                else:
                    print(f"  ⚠️ No text detected on page {page_num + 1}")
            
            doc.close()
            
            # Combine all pages into one big document
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
            
            # Calculate average confidence
            avg_confidence = sum(all_confidence_scores) / len(all_confidence_scores) if all_confidence_scores else 0
            
            print(f"✅ OCR Complete: Extracted {len(combined_text)} total characters from {total_pages} pages")
            print(f"   Average confidence: {avg_confidence:.2f}")
            
            if combined_text:
                print(f"TEXT PREVIEW: {combined_text[:100]}...")
            else:
                print("⚠️ No text detected in entire PDF")
            
            # Return ONE document with all pages combined
            return [Document(
                page_content=combined_text,
                metadata={
                    "source": os.path.basename(self.file_path),
                    "total_pages": total_pages,
                    "ocr_confidence": avg_confidence,
                    "extraction_method": "google_vision_api",
                    "original_format": ".pdf"
                }
            )]
            
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            raise