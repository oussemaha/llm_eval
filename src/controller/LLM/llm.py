


from http import client


class LLM:
    def __init__(self,vision_model:str,api_key:str=None,base_url:str=None):
        self.vision_model = vision_model
        self.client = client.OpenAI(
            api_key=api_key, 
            base_url=base_url
        )  

        
    def call_vision(self,image_path:str,user_prompt:str,MAX_TOKENS:int=150,temperature:float=0.1,system_prompt:str="You are a helpful assistant for analyzing medical documents."):
        
        response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_path},
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=temperature,
            )
        response = response.choices[0].message.content.strip()
        return response
    def call_text(self,user_prompt:str,MAX_TOKENS:int=150,temperature:float=0.1,system_prompt:str="You are a helpful assistant for analyzing medical documents."):
        
        response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=temperature,
            )
        response = response.choices[0].message.content.strip()
        return response