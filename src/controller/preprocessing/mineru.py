import requests
import os

class MineruClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/file_parse"
    def extract_from_response(self,response:dict):
        content=[]
        n_doc=0
        for item in response:
            content.append({"type": "text", "text": f"Document {n_doc} | {item} "})
            n_doc+=1
            print(item)
            elements=response[item]["md_content"].split("![]")
            for i in elements:
                if i.startswith("(images/"):
                    image_path = i[8:].split(")")[0]
                    try:
                        text = i[len(image_path)+10:]
                    except:
                        text = None
                    content.append({"type": "image_url", "image_url": {"url": response[item]["images"][image_path]}})
                    if text:
                        content.append({"type": "text", "text": text})
                else:
                    content.append({"type": "text", "text": i})
        return content

    def parse_file(self, files_list:list[str], backend="hybrid-http-client", 
                   parse_method="auto", server_url="http://127.0.0.1:31000", 
                   return_images=True):
        """
        Calls the /file_parse endpoint with specified parameters.
        """
        if len(files_list)==0 :
            raise FileNotFoundError(f"File not found: {file_path}")

        # The 'files' parameter in requests handles multipart/form-data
        files=[]
        for file_path in files_list:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            files.append(
                 (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')
            )
        files = {
            'files': files
        }

        # Non-file parameters are sent as data fields
        data = {
            'backend': backend,
            'parse_method': parse_method,
            'server_url': server_url,
            'return_images': str(return_images).lower(), # Convert boolean to 'true'/'false'
            'formula_enable': 'true', # Defaulting based on your screenshot
            'table_enable': 'true',
            'return_md': 'true'
        }

        try:
            response = requests.post(self.endpoint, files=files, data=data)
            response.raise_for_status()
            response=response.json()
            response=response["results"]
            return self.extract_from_response(response)
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "details": response.text if 'response' in locals() else None}
        finally:
            for file in files['files']:
                file[1].close()

# --- Example Usage ---
if __name__ == "__main__":
    client = FileParserClient(base_url="http://your-api-gateway.com")
    
    result = client.parse_file(
        files_list=["my_document.pdf"],
        backend="hybrid-http-client",
        parse_method="ocr",
        return_images=True
    )
    
    print(result)