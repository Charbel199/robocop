from fastapi import APIRouter
from starlette.responses import Response
from core.text_to_speech.text_to_speech_service import tts

router = APIRouter()


@router.post("")
def text_to_speach(text, stability=0.2, similarity=1):
    tts(text, stability=stability, similarity_boost=similarity)
    return Response(status_code=200)
