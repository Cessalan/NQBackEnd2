import os
import tempfile
from google.cloud import vision
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List
from PIL import Image

class OCRImageLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
        print("inside the image loader")

    def _convert_heic_to_jpg(self, heic_path: str) -> str:
        """Convert HEIC to JPG format for Google Vision"""
        try:
            # Register HEIC support
            from pillow_heif import register_heif_opener
            register_heif_opener()
            
            print(f"Converting HEIC file: {heic_path}")
            
            # Open and convert HEIC
            with Image.open(heic_path) as img:
                print(f"Original image mode: {img.mode}, size: {img.size}")
                
                # Convert to RGB
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                    print("Converted to RGB mode")
                
                # Create temporary JPG file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    # Save with high quality
                    img.save(tmp.name, 'JPEG', quality=95, optimize=True)
                    print(f"Saved converted image to: {tmp.name}")
                    
                    # Verify the saved file
                    if os.path.getsize(tmp.name) == 0:
                        raise Exception("Converted file is empty")
                    
                    return tmp.name
                    
        except ImportError:
            raise Exception("pillow-heif package required for HEIC files. Install: pip install pillow-heif")
        except Exception as e:
            raise Exception(f"HEIC conversion failed: {str(e)}")

    def load(self) -> List[Document]:
        """Extract text from image using Google Vision API"""
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("GOOGLE_API_KEY environment variable not set")
        
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        print("Google Vision client loaded")
        
        # Handle HEIC files
        file_to_process = self.file_path
        temp_converted_file = None
        
        if self.file_path.lower().endswith('.heic'):
            temp_converted_file = self._convert_heic_to_jpg(self.file_path)
            file_to_process = temp_converted_file
            print(f"Using converted file: {file_to_process}")
        
        try:
            # Verify file exists and has content
            if not os.path.exists(file_to_process):
                raise Exception(f"File not found: {file_to_process}")
            
            file_size = os.path.getsize(file_to_process)
            if file_size == 0:
                raise Exception("Image file is empty")
            
            print(f"Processing file: {file_to_process} (size: {file_size} bytes)")
            
            # Read image file
            with open(file_to_process, 'rb') as image_file:
                content = image_file.read()
            
            if len(content) == 0:
                raise Exception("No content read from image file")
            
            print(f"Read {len(content)} bytes from image")
            
            # Create Vision API image object
            image = vision.Image(content=content)
            
            # Use document_text_detection for handwriting
            response = client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Vision API Error: {response.error.message}")
            
            # Extract text
            text = ""
            if response.full_text_annotation:
                text = response.full_text_annotation.text
            
            # Calculate confidence
            confidence_scores = []
            if response.full_text_annotation and response.full_text_annotation.pages:
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        if hasattr(block, 'confidence'):
                            confidence_scores.append(block.confidence)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            print(f"✅ Extracted {len(text)} characters with {avg_confidence:.2f} confidence")
            if text:
                print(f"TEXT PREVIEW: {text[:100]}...")
            else:
                print("⚠️ No text detected in image")
            
            return [Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(self.file_path),
                    "ocr_confidence": avg_confidence,
                    "extraction_method": "google_vision_api",
                    "original_format": os.path.splitext(self.file_path)[-1].lower()
                }
            )]
            
        finally:
            # Clean up converted file
            if temp_converted_file and os.path.exists(temp_converted_file):
                os.unlink(temp_converted_file)
                print(f"Cleaned up temp file: {temp_converted_file}")