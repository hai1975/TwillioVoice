"""FastAPI server — Twilio Voice + Google Gemini."""

import logging
import os
from xml.sax.saxutils import escape

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Response, WebSocket
from fastapi.responses import JSONResponse
from google import genai
from twilio.rest import Client as TwilioClient

from conversation_relay import ConversationRelayHandler
from twilio_handler import TwilioHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Twilio Voice ⇄ Gemini",
    description="VoiceBot điện thoại với Google Gemini",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
LIVE_MODEL = os.getenv("MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
VOICE_MODE = os.getenv("VOICE_MODE", "conversationrelay").lower()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST", "localhost:8000")

SYSTEM_INSTRUCTION = os.getenv(
    "SYSTEM_INSTRUCTION",
    "You are the friendly phone receptionist for VM Clinic. "
    "Answer in clear English, keep responses short and natural for a phone call. "
    "Do not use special characters, bullet points, or emojis.",
)
GREETING = os.getenv(
    "GREETING",
    "Thanks for calling VM Clinic. Can I help you today?",
)
VOICE_NAME = os.getenv("VOICE_NAME", "Puck")
CR_LANGUAGE = os.getenv("CR_LANGUAGE", "en-US")
CR_VOICE = os.getenv("CR_VOICE", "UgBBYS2sOqTuMpoF3BR0")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def _host() -> str:
    return TWILIO_APP_HOST.replace("https://", "").replace("http://", "")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "twilliovoice",
        "mode": VOICE_MODE,
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": VOICE_MODE,
        "chat_model": CHAT_MODEL,
        "live_model": LIVE_MODEL,
        "language": CR_LANGUAGE,
        "voice": CR_VOICE,
        "greeting": GREETING[:80],
    }


@app.post("/twilio/inbound")
async def twilio_inbound():
    """Webhook cuộc gọi đến."""
    host = _host()

    if VOICE_MODE == "live":
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/twilio/stream" />
    </Connect>
</Response>"""
    else:
        greeting = escape(GREETING)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <ConversationRelay
            url="wss://{host}/twilio/relay"
            welcomeGreeting="{greeting}"
            language="{CR_LANGUAGE}"
            ttsProvider="ElevenLabs"
            voice="{CR_VOICE}"
        />
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio/relay")
async def twilio_relay(websocket: WebSocket):
    """ConversationRelay WebSocket — Twilio STT/TTS + Gemini text."""
    if not gemini_client:
        await websocket.close(code=1011, reason="GEMINI_API_KEY chưa cấu hình")
        return

    handler = ConversationRelayHandler(
        client=gemini_client,
        model=CHAT_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    await handler.handle(websocket)


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """Media Streams WebSocket — Gemini Live native audio (experimental)."""
    await websocket.accept()

    if not GEMINI_API_KEY:
        await websocket.close(code=1011, reason="GEMINI_API_KEY chưa cấu hình")
        return

    handler = TwilioHandler(
        gemini_api_key=GEMINI_API_KEY,
        model=LIVE_MODEL,
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
    to_number: str = Query(...),
    from_number: str = Query(...),
):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN]):
        return JSONResponse(status_code=500, content={"error": "Thiếu Twilio credentials"})

    host = _host()
    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    if VOICE_MODE == "live":
        inner = f'<Stream url="wss://{host}/twilio/stream" />'
    else:
        greeting = escape(GREETING)
        inner = f"""<ConversationRelay
            url="wss://{host}/twilio/relay"
            welcomeGreeting="{greeting}"
            language="{CR_LANGUAGE}"
            ttsProvider="ElevenLabs"
            voice="{CR_VOICE}"
        />"""

    twiml = f"<Response><Connect>{inner}</Connect></Response>"
    call = client.calls.create(to=to_number, from_=from_number, twiml=twiml)
    return {"callSid": call.sid, "status": call.status}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
