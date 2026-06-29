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
MULAW_SILENCE = b"\xff" * MULAW_FRAME_SIZE


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
        call_active = True
        gemini_audio_started = False
        media_in_count = 0
        media_out_count = 0
        silence_task: asyncio.Task | None = None

        async def send_mulaw_frame(frame: bytes):
            nonlocal media_out_count
            if not self.stream_sid:
                return
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
            media_out_count += 1

        async def send_buffered_audio():
            while len(output_buffer) >= MULAW_FRAME_SIZE:
                frame = bytes(output_buffer[:MULAW_FRAME_SIZE])
                del output_buffer[:MULAW_FRAME_SIZE]
                await send_mulaw_frame(frame)

        async def flush_output_buffer():
            await send_buffered_audio()

        async def silence_pumper():
            """Gửi silence tới Twilio trong lúc chờ Gemini phản hồi."""
            nonlocal gemini_audio_started
            while call_active and self.stream_sid and not gemini_audio_started:
                await send_mulaw_frame(MULAW_SILENCE)
                await asyncio.sleep(0.02)

        async def audio_output_callback(data: bytes):
            nonlocal gemini_audio_started
            gemini_audio_started = True

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
            nonlocal gemini_audio_started
            output_buffer.clear()
            self._downsample_state_1 = None
            self._downsample_state_2 = None
            gemini_audio_started = False
            if self.stream_sid:
                await websocket.send_text(
                    json.dumps({"event": "clear", "streamSid": self.stream_sid})
                )

        # Kết nối Gemini ngay khi WebSocket mở — song song với Twilio setup
        gemini_task = asyncio.create_task(
            self.gemini_client.start_session(
                audio_input_queue=audio_input_queue,
                text_input_queue=text_input_queue,
                audio_output_callback=audio_output_callback,
                audio_interrupt_callback=audio_interrupt_callback,
            )
        )
        greeting_sent = False

        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                event = data.get("event")
                logger.info("Twilio event: %s", event)

                if event == "connected":
                    continue

                if event == "start":
                    self.stream_sid = data["start"]["streamSid"]
                    logger.info("Stream bắt đầu: %s", self.stream_sid)
                    await flush_output_buffer()
                    silence_task = asyncio.create_task(silence_pumper())
                    if not greeting_sent:
                        await text_input_queue.put(self.greeting)
                        greeting_sent = True

                elif event == "media":
                    media_in_count += 1
                    payload = data["media"]["payload"]
                    mulaw_data = base64.b64decode(payload)
                    pcm_data = audioop.ulaw2lin(mulaw_data, 2)
                    resampled_data, self._upsample_state = audioop.ratecv(
                        pcm_data, 2, 1, 8000, 16000, self._upsample_state
                    )
                    await audio_input_queue.put(resampled_data)

                elif event == "stop":
                    logger.info(
                        "Stream kết thúc: %s (media_in=%d, media_out=%d)",
                        self.stream_sid,
                        media_in_count,
                        media_out_count,
                    )
                    break

        except Exception as e:
            logger.error("Lỗi xử lý Twilio stream: %s", e, exc_info=True)
        finally:
            call_active = False
            if silence_task:
                silence_task.cancel()
            await audio_input_queue.put(None)
            await text_input_queue.put(None)
            gemini_task.cancel()
            try:
                await gemini_task
            except asyncio.CancelledError:
                pass
