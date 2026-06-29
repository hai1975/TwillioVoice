"""Xử lý Twilio Media Streams — chuyển đổi audio và bridge với Gemini Live."""

import asyncio
import base64
import json
import logging

try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # noqa: F401 — Python 3.13+

from fastapi import WebSocket

from gemini_live import GeminiLive

logger = logging.getLogger(__name__)

MULAW_FRAME_SIZE = 160  # 20ms tại 8kHz


class TwilioHandler:
    def __init__(
        self,
        gemini_api_key: str,
        model: str,
        system_instruction: str = "Bạn là trợ lý AI thân thiện, trả lời bằng tiếng Việt.",
        voice_name: str = "Puck",
        greeting: str = "Chào người gọi và hỏi bạn có thể giúp gì.",
    ):
        self.gemini_client = GeminiLive(
            api_key=gemini_api_key,
            model=model,
            system_instruction=system_instruction,
            voice_name=voice_name,
        )
        self.greeting = greeting
        self.stream_sid: str | None = None
        self._upsample_state = None
        self._downsample_state_1 = None
        self._downsample_state_2 = None

    async def handle_media_stream(self, websocket: WebSocket):
        audio_input_queue: asyncio.Queue = asyncio.Queue()
        text_input_queue: asyncio.Queue = asyncio.Queue()
        output_buffer = bytearray()
        gemini_task: asyncio.Task | None = None

        async def send_buffered_audio():
            nonlocal output_buffer
            while len(output_buffer) >= MULAW_FRAME_SIZE:
                frame = bytes(output_buffer[:MULAW_FRAME_SIZE])
                del output_buffer[:MULAW_FRAME_SIZE]
                payload = base64.b64encode(frame).decode("utf-8")
                await websocket.send_text(
                    json.dumps(
                        {
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": payload},
                        }
                    )
                )

        async def audio_output_callback(data: bytes):
            if not self.stream_sid:
                return

            intermediate, self._downsample_state_1 = audioop.ratecv(
                data, 2, 1, 24000, 16000, self._downsample_state_1
            )
            resampled, self._downsample_state_2 = audioop.ratecv(
                intermediate, 2, 1, 16000, 8000, self._downsample_state_2
            )
            mulaw_data = audioop.lin2ulaw(resampled, 2)
            output_buffer.extend(mulaw_data)
            await send_buffered_audio()

        async def audio_interrupt_callback():
            nonlocal output_buffer
            output_buffer.clear()
            self._downsample_state_1 = None
            self._downsample_state_2 = None
            if self.stream_sid:
                await websocket.send_text(
                    json.dumps({"event": "clear", "streamSid": self.stream_sid})
                )

        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                event = data.get("event")

                if event == "start":
                    self.stream_sid = data["start"]["streamSid"]
                    logger.info("Stream bắt đầu: %s", self.stream_sid)

                    # Chờ event "start" (có streamSid) rồi mới kết nối Gemini
                    gemini_task = asyncio.create_task(
                        self.gemini_client.start_session(
                            audio_input_queue=audio_input_queue,
                            text_input_queue=text_input_queue,
                            audio_output_callback=audio_output_callback,
                            audio_interrupt_callback=audio_interrupt_callback,
                        )
                    )
                    await text_input_queue.put(self.greeting)

                elif event == "media":
                    payload = data["media"]["payload"]
                    mulaw_data = base64.b64decode(payload)
                    pcm_data = audioop.ulaw2lin(mulaw_data, 2)
                    resampled_data, self._upsample_state = audioop.ratecv(
                        pcm_data, 2, 1, 8000, 16000, self._upsample_state
                    )
                    await audio_input_queue.put(resampled_data)

                elif event == "stop":
                    logger.info("Stream kết thúc: %s", self.stream_sid)
                    break

        except Exception as e:
            logger.error("Lỗi xử lý Twilio stream: %s", e, exc_info=True)
        finally:
            await audio_input_queue.put(None)
            await text_input_queue.put(None)
            if gemini_task:
                gemini_task.cancel()
                try:
                    await gemini_task
                except asyncio.CancelledError:
                    pass
