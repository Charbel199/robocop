import requests
import os

voice_id = 'jezrhkF9hm5kXYpbND6B'
url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
api_key = os.getenv('API_KEY', default=None)

# Define the headers
headers = {
    'accept': 'audio/mpeg',
    'xi-api-key': api_key,
    'Content-Type': 'application/json'
}


def tts(text, stability=0.3, similarity_boost=1, output_path='./temp.mp3'):
    # Define the payload
    payload = {
        'text': text,
        'voice_settings': {
            'stability': stability,
            'similarity_boost': similarity_boost
        }
    }
    # Make the HTTP request
    response = requests.post(url, headers=headers, json=payload)

    # Check the response status code
    if response.status_code == 200:
        # Do something with the response content, e.g. write it to a file
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        # Handle the error
        print(f'Request failed with status code {response.status_code}: {response.text}')