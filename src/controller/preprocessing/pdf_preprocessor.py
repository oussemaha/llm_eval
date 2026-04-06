import logging

logger = logging.getLogger(__name__)
import PyPDF2
from PIL import Image
import pdf2image
from pathlib import Path
from typing import List, Tuple, Union
import io


class PDFPreprocessor:
    def __init__(self):
        pass

    def is_editable_pdf(self, pdf_path: str) -> bool:
        """Check if PDF is editable (contains extractable text)."""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking PDF: {e}")
            return False

    def extract_text_and_images(self, pdf_path: str) -> Tuple[str, List[Image.Image]]:
        """Extract text and images from editable PDF."""
        text = ""
        images = []

        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                    # Extract images from page
                    if "/Resources" in page and "/XObject" in page["/Resources"]:
                        xObject = page["/Resources"]["/XObject"].get_object()
                        for obj in xObject:
                            if xObject[obj]["/Subtype"] == "/Image":
                                try:
                                    image_obj = xObject[obj]

                                    # Skip soft masks and non-renderable images
                                    if image_obj.get("/ImageMask", False):
                                        continue

                                    image_data = image_obj.get_data()
                                    filter_type = image_obj.get("/Filter", "")

                                    # Handle chained filters (filter can be a list like [/FlateDecode])
                                    if isinstance(filter_type, list):
                                        filter_type = (
                                            filter_type[0] if filter_type else ""
                                        )

                                    # Handle different PDF image encodings
                                    if filter_type == "/DCTDecode":
                                        # JPEG image
                                        image = Image.open(io.BytesIO(image_data))
                                    elif filter_type == "/FlateDecode":
                                        # Zlib compressed image (often PNG-like)
                                        width = image_obj["/Width"]
                                        height = image_obj["/Height"]
                                        color_space = image_obj.get(
                                            "/ColorSpace", "/DeviceRGB"
                                        )
                                        bits_per_component = image_obj.get(
                                            "/BitsPerComponent", 8
                                        )

                                        # Safely handle if color_space is an ArrayObject or list
                                        cs_name = color_space
                                        if isinstance(color_space, list):
                                            cs_name = (
                                                color_space[0]
                                                if color_space
                                                else "/DeviceRGB"
                                            )

                                        mode_map = {
                                            "/DeviceRGB": ("RGB", 3),
                                            "/DeviceGray": ("L", 1),
                                            "/DeviceCMYK": ("CMYK", 4),
                                        }

                                        mode, channels = mode_map.get(
                                            str(cs_name), ("RGB", 3)
                                        )
                                        expected_len = (
                                            width
                                            * height
                                            * channels
                                            * bits_per_component
                                        ) // 8

                                        if len(image_data) >= expected_len:
                                            try:
                                                image = Image.frombytes(
                                                    mode,
                                                    (width, height),
                                                    image_data[:expected_len],
                                                )
                                            except Exception:
                                                image = Image.open(
                                                    io.BytesIO(image_data)
                                                )
                                        else:
                                            image = Image.open(io.BytesIO(image_data))
                                    else:
                                        # Fallback to generic opening
                                        image = Image.open(io.BytesIO(image_data))

                                    images.append(image)
                                except Exception:
                                    pass  # Silently skip undecodeable images (masks, indexed colors, etc.)

        except Exception as e:
            logger.error(f"Error extracting content: {e}")

        return text, images

    def extract_pages_as_images(self, pdf_path: str) -> List[Image.Image]:
        """Extract each page as an image."""
        try:
            images = pdf2image.convert_from_path(pdf_path)

            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []

    def preprocess(
        self, pdf_path: str
    ) -> Union[Tuple[str, List[Image.Image]], List[Image.Image]]:
        """Main preprocessing function."""
        if self.is_editable_pdf(pdf_path):
            return self.extract_text_and_images(pdf_path)
        else:
            return self.extract_pages_as_images(pdf_path)


if __name__ == "__main__":
    import sys

    # Test path
    test_pdf = "/home/oussema/Downloads/23.pdf"

    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]

    preprocessor = PDFPreprocessor()

    print(f"--- Testing PDFPreprocessor with: {test_pdf} ---")

    if not Path(test_pdf).exists():
        print(f"File not found: {test_pdf}")
        print("Please provide a valid PDF file path as an argument.")
    else:
        print(f"Checking if {test_pdf} is editable...")
        is_editable = preprocessor.is_editable_pdf(test_pdf)
        print(f"Is editable: {is_editable}")

        print("\nProcessing PDF...")
        result = preprocessor.preprocess(test_pdf)

        if isinstance(result, tuple):
            text, images = result
            print(f"Extracted Text (first 100 chars): {text}...")
            print(f"Extracted Images count: {len(images)} ")

            # Save images for inspection
            out_dir = Path("output_images")
            out_dir.mkdir(exist_ok=True)
            for i, img in enumerate(images):
                out_path = out_dir / f"image_{i}.png"
                img.save(out_path)
            print(f"Saved {len(images)} images to '{out_dir}/'")
        else:
            print(f"Extracted Pages as Images count: {len(result)}")

            # Save images for inspection
            out_dir = Path("output_images")
            out_dir.mkdir(exist_ok=True)
            for i, img in enumerate(result):
                out_path = out_dir / f"page_{i}.png"
                img.save(out_path)
            print(f"Saved {len(result)} page images to '{out_dir}/'")

        print("\n--- Testing Complete ---")
