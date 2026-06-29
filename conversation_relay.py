"""Twilio ConversationRelay — Twilio STT/TTS + Gemini Chat (text)."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from google import genai

from patient_registration import REGISTRATION_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

REGISTRATIONS_DIR = Path("data/registrations")


class ConversationRelayHandler:
    def __init__(
        self,
        client: genai.Client,
        model: str,
        system_instruction: str,
        extraction_prompt: str = REGISTRATION_EXTRACTION_PROMPT,
    ):
        self.client = client
        self.model = model
        self.system_instruction = system_instruction
        self.extraction_prompt = extraction_prompt
        self.sessions: dict[str, object] = {}
        self._saved_calls: set[str] = set()

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
                    reply = response.text or ""
                    logger.info("Gemini trả lời: %s", reply[:120])

                    if "REGISTRATION_COMPLETE" in reply:
                        reply = reply.replace("REGISTRATION_COMPLETE", "").strip()
                        await self._save_registration(chat, call_sid)

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
                chat = self.sessions.pop(call_sid)
                await self._save_registration(chat, call_sid)

    async def _save_registration(self, chat, call_sid: str):
        if call_sid in self._saved_calls:
            return
        self._saved_calls.add(call_sid)
        try:
            response = chat.send_message(self.extraction_prompt)
            raw = (response.text or "").strip()
            match = re.search(r"\{[\s\S]*\}", raw)
            if not match:
                logger.warning("Không trích xuất được JSON cho call %s", call_sid)
                return

            data = json.loads(match.group())
            data["call_sid"] = call_sid
            data["saved_at"] = datetime.now(timezone.utc).isoformat()

            REGISTRATIONS_DIR.mkdir(parents=True, exist_ok=True)
            path = REGISTRATIONS_DIR / f"{call_sid}.json"
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Đã lưu đăng ký: %s — %s", path, data.get("patient_name"))
        except Exception as e:
            logger.error("Lỗi lưu đăng ký call %s: %s", call_sid, e, exc_info=True)
