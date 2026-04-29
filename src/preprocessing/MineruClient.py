import requests
import os
import mimetypes
import logging
import json

logger = logging.getLogger(__name__)

def extract_unique_images(data, image_list=None):
    if image_list is None:
        image_list = {}

    if isinstance(data, dict):
        url = data.get("image_path")
        if url and url not in image_list:
            image_list[url] = data.get("type", "unknown")
        
        for value in data.values():
            extract_unique_images(value,  image_list)
            
    elif isinstance(data, list):
        for item in data:
            extract_unique_images(item,  image_list)
            
    return image_list

class MineruClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/file_parse"
    def extract_unique_images(self,data, image_list=None):
        if image_list is None:
            image_list = {}

        if isinstance(data, dict):
            url = data.get("image_path")
            if url and url not in image_list:
                image_list[url] = data.get("type", "unknown")

            for value in data.values():
                self.extract_unique_images(value,  image_list)

        elif isinstance(data, list):
            for item in data:
                self.extract_unique_images(item,  image_list)

        return image_list

    def extract_from_response(self,response:dict,return_images:bool):
        content=[]
        content_md=[]


        n_doc=0
        for item in response:
            content.append({"type": "text", "text": f"Document {n_doc} | {item} "})
            n_doc+=1
            if not return_images:
                content.append({"type": "text", "text": response[item]["md_content"]})
                continue
            elements=response[item]["md_content"].split("![]")
            images_type=self.extract_unique_images(json.loads(response[item]["middle_json"]))
            for i in elements:
                if i.startswith("(images/"):
                    image_path = i[8:].split(")")[0]
                    try:
                        text = i[len(image_path)+10:]
                    except:
                        text = None
                    content.append({"type": "image_url", "image_url": {"url": response[item]["images"][image_path],"type":images_type[image_path]}})
                    if text:
                        content.append({"type": "text", "text": text})
                        content_md.append(text)
                else:
                    content.append({"type": "text", "text": i})
                    content_md.append({"type": "text", "text": i})
        return content

    def parse_file(self, files_list:list[str], **kwargs):
        """
        Calls the /file_parse endpoint with specified parameters.
        """
        data={}
        data["lang_list"]=kwargs.get("lang_list","latin")
        data["backend"]=kwargs.get("backend","pipeline")
        data["parse_method"]=kwargs.get("parse_method","auto")
        data["server_url"]=kwargs.get("server_url","")
        data["return_images"]=str(kwargs.get("return_images",True)).lower()
        data["formula_enable"]=str(kwargs.get("formula_enable",False)).lower()
        data["table_enable"]=str(kwargs.get("table_enable",False)).lower()
        data["return_md"]=str(kwargs.get("return_md",True)).lower()
        data["return_middle_json"]=data["return_images"]
        
        if len(files_list)==0 :
            raise FileNotFoundError(f"File not found: {file_path}")

        # The 'files' parameter in requests handles multipart/form-data
        files_to_send = []
        image_paths = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
        
        for file_path in files_list:
                if file_path.lower().endswith(image_extensions):
                    image_paths.append(file_path)

                if not os.path.exists(file_path):
                    logger.warning(f"Warning: File not found {file_path}")
                    continue
                
                # Auto-detect mime type (application/pdf, image/jpeg, etc.)
                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or 'application/octet-stream'
                
                f = open(file_path, 'rb')
                
                files_to_send.append(
                    ('files', (os.path.basename(file_path), f, mime_type))
                )


        try:
            response = requests.post(self.endpoint, files=files_to_send, data=data)
            response.raise_for_status()
            response=response.json()
            response=response["results"]
            response=self.extract_from_response(response,return_images=kwargs.get("return_images",True))


            return response
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "details": response.text if 'response' in locals() else None}


# --- Example Usage --- 
if __name__ == "__main__":
    client = MineruClient()
    
    result = client.parse_file(
        files_list=["/home/oussema/Downloads/Downloads/pdf_test/fiche_de_stim.png","/home/oussema/Downloads/Downloads/pdf_test/1.pdf"],
        return_images=True    )
    print(result)
