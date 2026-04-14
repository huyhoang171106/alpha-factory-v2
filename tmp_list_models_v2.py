
import requests
import os
import json

url = "https://ollama.com/api/tags"
headers = {"Authorization": "Bearer 81112fa746b54980a47d192a7e98c05d.kzBt2JsGrcoemsyJ314oBS_7"}

try:
    response = requests.get(url, headers=headers)
    result = {
        "status": response.status_code,
        "body": response.json() if response.status_code == 200 else response.text
    }
    with open("d:/alpha-factory-private/tmp_list_models.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Done")
except Exception as e:
    print(f"Error: {e}")
