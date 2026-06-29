"""Twilio ConversationRelay — Twilio STT/TTS + Gemini Chat (text)."""

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from google import genai

logger = logging.getLogger(__name__)


class ConversationRelayHandler:
    def __init__(
        self,
        client: genai.Client,
        model: str,
        system_instruction: str,
    ):
        self.client = client
        self.model = model
        self.system_instruction = system_instruction
        self.sessions: dict[str, object] = {}

    async def handle(self, websocket: WebSocket):
        await websocket.accept()
        call_sid: str | None = None

        try:
            while True:
                raw = await websocket.receive_text()
                message = json.loads(raw)
                msg_type = message.get("type")
                logger.info("ConversationRelay: %s", msg_type)

                if msg_type == "setup":
                    call_sid = message["callSid"]
                    self.sessions[call_sid] = self.client.chats.create(
                        model=self.model,
                        config={"system_instruction": self.system_instruction},
                    )
                    logger.info("Session tạo cho call %s", call_sid)

                elif msg_type == "prompt":
                    if not call_sid or call_sid not in self.sessions:
                        logger.error("Prompt không có session: %s", call_sid)
                        continue

                    user_text = message.get("voicePrompt", "")
                    logger.info("User nói: %s", user_text)

                    chat = self.sessions[call_sid]
                    response = chat.send_message(user_text)
                    reply = response.text
                    logger.info("Gemini trả lời: %s", reply[:120])

                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "text",
                                "token": reply,
                                "last": True,
                            }
                        )
                    )

                elif msg_type == "interrupt":
                    logger.info("User ngắt lời call %s", call_sid)

        except WebSocketDisconnect:
            logger.info("WebSocket đóng call %s", call_sid)
        except Exception as e:
            logger.error("ConversationRelay lỗi: %s", e, exc_info=True)
        finally:
            if call_sid and call_sid in self.sessions:
                self.sessions.pop(call_sid)
