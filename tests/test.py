import base64
import os
import csv
import json
from openai import OpenAI
from preprocesing import load_prompts, build_graph

class Test:
    def __init__(self):
        self.client = OpenAI()

    def vision_judge(self,image_path, model_output):

        image_b64 = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""
                            You are a medical document expert.

                            Here is a model prediction:
                            {model_output}

                            Evaluate:
                            1. Is the predicted category correct?
                            2. Is the description faithful to the image?
                            3. Does it hallucinate?

                            Return ONLY JSON:

                            {{
                              "classification_score": 0-1,
                              "description_score": 0-1,
                              "hallucination_score": 0-1
                            }}
                            """
                        }
                    ]
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

if __name__ == "__main__":
    test = Test()
    load_prompts()
    app = build_graph()
    
    directory = "/home/oussema/Downloads"
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                images.append(os.path.join(root, file))
    
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'doc_type', 'confidence', 'doc_desc', 'classification_score', 'description_score', 'hallucination_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for image_path in images:
            initial_state = {
                "image_path": image_path,
                "image_b64": "",
                "doc_type": "unknown",
                "confidence": "",
                "doc_desc": "",
            }
            final_state = app.invoke(initial_state)
            model_output = final_state['doc_desc']
            judge_result_str = test.vision_judge(image_path, model_output)
            try:
                judge_result = json.loads(judge_result_str)
            except json.JSONDecodeError:
                judge_result = {"classification_score": None, "description_score": None, "hallucination_score": None}
            writer.writerow({
                'image_path': image_path,
                'doc_type': final_state['doc_type'],
                'confidence': final_state['confidence'],
                'doc_desc': final_state['doc_desc'],
                'classification_score': judge_result.get('classification_score'),
                'description_score': judge_result.get('description_score'),
                'hallucination_score': judge_result.get('hallucination_score'),
            })