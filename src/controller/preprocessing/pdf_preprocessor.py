import PyPDF2
from PIL import Image
import pdf2image
from pathlib import Path
from typing import List, Tuple, Union

class PDFPreprocessor:
    def __init__(self):
        pass

    def is_editable_pdf(self,pdf_path: str) -> bool:
        """Check if PDF is editable (contains extractable text)."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        return True
            return False
        except Exception as e:
            print(f"Error checking PDF: {e}")
            return False
    
    def extract_text_and_images(self,pdf_path: str) -> Tuple[str, List[Image.Image]]:
        """Extract text and images from editable PDF."""
        text = ""
        images = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                    
                    # Extract images from page
                    if "/XObject" in page["/Resources"]:
                        xObject = page["/Resources"]["/XObject"].get_object()
                        for obj in xObject:
                            if xObject[obj]["/Subtype"] == "/Image":
                                try:
                                    image = Image.open(xObject[obj].get_data())
                                    images.append(image)
                                except Exception as e:
                                    print(f"Error extracting image: {e}")
        except Exception as e:
            print(f"Error extracting content: {e}")
        
        return text, images
    
    def extract_pages_as_images(self,pdf_path: str) -> List[Image.Image]:
        """Extract each page as an image."""
        try:
            images = pdf2image.convert_from_path(pdf_path)
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []
    
    def preprocess(self, pdf_path: str) -> Union[Tuple[str, List[Image.Image]], List[Image.Image]]:
        """Main preprocessing function."""
        if self.is_editable_pdf(pdf_path):
            return self.extract_text_and_images()
        else:
            return self.extract_pages_as_images()