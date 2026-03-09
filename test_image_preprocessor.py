import numpy as np
from PIL import Image
from src.controller.preprocessing.Image_Preprocessor import ImagePreprocessor

# Create a test JPEG
img_jpeg = Image.new('RGB', (10, 10), color='red')
img_jpeg.save('test.jpg', format='JPEG')

# Create a test PNG
img_png = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))
img_png.save('test.png', format='PNG')

processor = ImagePreprocessor()

try:
    print("Testing string path (JPEG):")
    res1 = processor.image_to_base64('test.jpg')
    print(res1[:50] + "...")
    assert res1.startswith("data:image/jpeg;base64,")
    
    print("Testing PIL Image (PNG):")
    res2 = processor.image_to_base64(img_png)
    print(res2[:50] + "...")
    assert res2.startswith("data:image/png;base64,")
    
    print("Testing Numpy Array (RGB):")
    arr = np.array(img_jpeg)
    res3 = processor.image_to_base64(arr)
    print(res3[:50] + "...")
    assert res3.startswith("data:image/png;base64,") # numpy defaults to PNG
    
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()

import os
os.remove('test.jpg')
os.remove('test.png')
