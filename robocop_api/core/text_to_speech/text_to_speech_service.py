import requests
import os
from pydub import AudioSegment
from pydub.playback import play
import io
voice_id = 'HdHijm07A1ZXRVGHVvjs'
url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
api_key = os.getenv('API_KEY', default=None)
if api_key is None:
    raise Exception("Env variable 'API_KEY' not set.")
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

    audio = AudioSegment.from_file(io.BytesIO(response.content), format='mp3')

    # Play the audio file
    play(audio)

    # Check the response status code
    if response.status_code == 200:
        # Do something with the response content, e.g. write it to a file
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        # Handle the error
        print(f'Request failed with status code {response.status_code}: {response.text}')

if __name__ == "__main__":
    tts(f"ATTENTION!!! Georges Daou \n\n this is the POLICE!!!. PULL OVER NOW OR WE WILL USE NECESSARY FORCE TO STOP YOUR VEHICLE !!!!! RATATATATATATA",
        similarity_boost=1,
        stability=0.6)