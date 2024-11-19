import requests
import json
import time

# Define the API endpoint
url = "http://ec2-43-198-104-59.ap-east-1.compute.amazonaws.com:11434/api/generate"
# url = "http://localhost:11434/api/generate"

headers = {"Content-Type": "application/json"}

prompt = "strictly only respond with the output: Please rephrase this sentence in clear English with no extra explanation or repetition: tum kya khaana chahoge?"

data = {"prompt": prompt}

start_time = time.time()
# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))
end_time = time.time()
latency = end_time - start_time
print(f"Latency: {latency} seconds")

print("LLM response: " + response.text)
