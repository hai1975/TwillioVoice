"""FastAPI server — bridge Twilio Media Streams ⇄ Gemini Live API."""

import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Response, WebSocket
from fastapi.responses import JSONResponse
from twilio.rest import Client as TwilioClient

from twilio_handler import TwilioHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Twilio Voice ⇄ Gemini Live",
    description="Cuộc gọi điện thoại 2 chiều với VoiceBot qua Gemini Live API",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("MODEL", "gemini-2.5-flash-live-preview")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST", "localhost:8000")
SYSTEM_INSTRUCTION = os.getenv(
    "SYSTEM_INSTRUCTION",
    "Bạn là trợ lý AI thân thiện. Trả lời bằng tiếng Việt, ngắn gọn và tự nhiên.",
)
VOICE_NAME = os.getenv("VOICE_NAME", "Puck")
GREETING = os.getenv(
    "GREETING",
    "Chào người gọi bằng tiếng Việt và hỏi bạn có thể giúp gì.",
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "twilliovoice", "health": "/health"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/twilio/inbound")
async def twilio_inbound():
    """Webhook khi có cuộc gọi đến số Twilio."""
    host = TWILIO_APP_HOST.replace("https://", "").replace("http://", "")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="en-US">Connecting you to the AI assistant.</Say>
    <Connect>
        <Stream url="wss://{host}/twilio/stream" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """WebSocket nhận audio stream từ Twilio."""
    await websocket.accept()

    if not GEMINI_API_KEY:
        await websocket.close(code=1011, reason="GEMINI_API_KEY chưa được cấu hình")
        return

    handler = TwilioHandler(
        gemini_api_key=GEMINI_API_KEY,
        model=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        voice_name=VOICE_NAME,
        greeting=GREETING,
    )
    try:
        await handler.handle_media_stream(websocket)
    except Exception as e:
        logger.error("Twilio stream error: %s", e, exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/twilio/outbound")
async def twilio_outbound(
    to_number: str = Query(..., description="Số điện thoại người nhận, VD: +84901234567"),
    from_number: str = Query(..., description="Số Twilio gọi đi, VD: +12025551234"),
):
    """Gọi ra ngoài và kết nối với Gemini Live (cần bảo mật trong production)."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN]):
        return JSONResponse(
            status_code=500,
            content={"error": "TWILIO_ACCOUNT_SID và TWILIO_AUTH_TOKEN chưa được cấu hình"},
        )

    host = TWILIO_APP_HOST.replace("https://", "").replace("http://", "")
    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    twiml = f"""<Response>
    <Say language="en-US">Connecting you to the AI assistant.</Say>
    <Connect>
        <Stream url="wss://{host}/twilio/stream" />
    </Connect>
</Response>"""

    call = client.calls.create(to=to_number, from_=from_number, twiml=twiml)
    return {"callSid": call.sid, "status": call.status}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
