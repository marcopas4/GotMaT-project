"""
OCR Extractor for PDF Documents
Extracts text from PDF files using PyMuPDF and Tesseract OCR.
Handles both text-based PDFs (direct extraction) and image-based PDFs (OCR extraction).
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
from pathlib import Path
import io

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """
    Extract text from PDF files using multiple methods:
    1. Direct text extraction (for text-based PDFs)
    2. OCR extraction (for image-based PDFs)
    """
    
    def __init__(self, input_dir, output_dir, dpi=300, ocr_threshold=50):
        """
        Initialize the PDFTextExtractor.
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory to save extracted text
            dpi (int): DPI for image conversion (default: 300)
            ocr_threshold (int): Min chars for direct extraction (default: 50)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.ocr_threshold = ocr_threshold
        
        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDFTextExtractor initialized (DPI: {dpi}, threshold: {ocr_threshold})")
    
    def preprocess_image(self, image):
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Grayscale conversion
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold (Otsu's method)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL
        return Image.fromarray(thresh)
    
    def extract_text_from_page(self, page, force_ocr=False):
        """
        Extract text from a single PDF page.
        
        Args:
            page (fitz.Page): PDF page object
            force_ocr (bool): Force OCR even if direct extraction works
            
        Returns:
            str: Extracted text
        """
        # Try direct text extraction
        text = page.get_text()
        
        # Use OCR if direct extraction yields little text or force_ocr is True
        if force_ocr or len(text.strip()) < self.ocr_threshold:
            try:
                # Convert page to image
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                
                # Preprocess and OCR
                img = self.preprocess_image(img)
                ocr_text = pytesseract.image_to_string(img,lang='ita')
                
                # Use OCR text if longer
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            
            except Exception as e:
                logger.warning(f"OCR failed for page, using direct extraction: {e}")
        
        return text
    
    def extract_text_from_pdf(self, pdf_path, force_ocr=False):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to PDF file
            force_ocr (bool): Force OCR for all pages
            
        Returns:
            dict: Dictionary with page numbers as keys and text as values
        """
        logger.info(f"Processing PDF: {pdf_path} (force_ocr={force_ocr})")
        
        try:
            doc = fitz.open(pdf_path)
            num_pages = doc.page_count
            
            pages_text = {}
            for page_num in range(num_pages):
                page = doc[page_num]
                text = self.extract_text_from_page(page, force_ocr=force_ocr)
                pages_text[page_num + 1] = text  # 1-based indexing
            
            doc.close()
            logger.info(f"Extracted text from {num_pages} pages")
            return pages_text
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {}
    
    