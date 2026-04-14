
import requests
import os

url = "https://ollama.com/api/tags"
headers = {"Authorization": "Bearer 81112fa746b54980a47d192a7e98c05d.kzBt2JsGrcoemsyJ314oBS_7"}

try:
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Body: {response.text}")
except Exception as e:
    print(f"Error: {e}")
