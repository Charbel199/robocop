from fastapi import APIRouter
from starlette.responses import Response
from core.text_to_speech.text_to_speech_service import tts
from fastapi import BackgroundTasks

router = APIRouter()


@router.post("")
async def text_to_speach(text, stability, similarity,background_tasks: BackgroundTasks):
    background_tasks.add_task(tts,text, stability=stability, similarity_boost=similarity)
    return Response(status_code=200)
