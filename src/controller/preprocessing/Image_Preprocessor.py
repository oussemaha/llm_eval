import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

class ImagePreprocessor:
    """Class for image preprocessing operations."""
    
    def __init__(self):
        pass
    
    def image_to_base64(self, image):
        """
        Convert an image to base64 encoded string with data URL prefix.
        
        Args:
            image: PIL Image, numpy array, or file path (str)
            
        Returns:
            str: Data URL with base64 encoded image
        """
        # Default format
        img_format = 'PNG'
        mime_type = 'image/png'

        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image)
            img_format = image.format or 'PNG'
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img_format = image.format or 'PNG'
            
        # Determine MIME type from format mapping
        format_to_mime = {
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'PNG': 'image/png',
            'GIF': 'image/gif',
            'BMP': 'image/bmp',
            'WEBP': 'image/webp'
        }
        mime_type = format_to_mime.get(img_format.upper(), 'image/png')
        
        # Convert to RGB to avoid issues saving JPEG with alpha channels
        if img_format.upper() in ['JPEG', 'JPG'] and image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        # Convert to bytes
        buffer = BytesIO()
        image.save(buffer, format=img_format)
        img_bytes = buffer.getvalue()
        
        # Encode to base64 with data URL prefix
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{base64_string}"
    
    def convert_to_grayscale(self, image):
        """
        Convert an image to grayscale.
        
        Args:
            image: PIL Image, numpy array, or file path (str)
            
        Returns:
            PIL Image or numpy array: Grayscale image
        """
        # Handle different input types
        if isinstance(image, str):
            image = cv2.imread(image)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, Image.Image):
            grayscale = image.convert('L')
        else:
            # Assume numpy array
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return grayscale