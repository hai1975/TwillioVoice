"""Quản lý session Gemini Live API — nhận audio từ queue, gửi output qua callback."""

import asyncio
import inspect
import logging
import traceback
from collections.abc import Callable

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SILENT_CHUNK_16K = b"\x00" * 640


class GeminiLive:
    def __init__(
        self,
        api_key: str,
        model: str,
        system_instruction: str = "Bạn là trợ lý AI thân thiện, trả lời bằng tiếng Việt.",
        voice_name: str = "Puck",
    ):
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction
        self.voice_name = voice_name
        self.client = genai.Client(api_key=api_key)

    async def start_session(
        self,
        audio_input_queue: asyncio.Queue,
        text_input_queue: asyncio.Queue,
        audio_output_callback,
        audio_interrupt_callback=None,
        is_active: Callable[[], bool] | None = None,
    ):
        active = is_active or (lambda: True)
        session_handle: str | None = None
        audio_sent_count = 0

        while active():
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice_name
                        )
                    )
                ),
                system_instruction=types.Content(
                    parts=[types.Part(text=self.system_instruction)]
                ),
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                session_resumption=types.SessionResumptionConfig(handle=session_handle),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        disabled=False,
                        start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                        end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                        prefix_padding_ms=300,
                        silence_duration_ms=500,
                    ),
                ),
            )

            try:
                async with self.client.aio.live.connect(
                    model=self.model, config=config
                ) as session:
                    logger.info(
                        "Gemini Live session đã kết nối (model=%s, resume=%s)",
                        self.model,
                        session_handle is not None,
                    )

                    async def send_audio_loop():
                        nonlocal audio_sent_count
                        while active():
                            try:
                                audio_data = await asyncio.wait_for(
                                    audio_input_queue.get(), timeout=0.1
                                )
                                if audio_data is None:
                                    continue
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=audio_data,
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                )
                                audio_sent_count += 1
                                if audio_sent_count in (1, 50, 100, 500, 1000):
                                    logger.info(
                                        "Đã gửi %d gói audio tới Gemini",
                                        audio_sent_count,
                                    )
                            except asyncio.TimeoutError:
                                continue
                            except Exception as e:
                                logger.error("Lỗi gửi audio: %s", e, exc_info=True)
                                break

                    async def send_text_loop():
                        while active():
                            try:
                                text = await asyncio.wait_for(
                                    text_input_queue.get(), timeout=0.1
                                )
                                if text is None:
                                    continue
                                logger.info("Gửi text prompt: %s", text[:80])
                                await session.send_realtime_input(text=text)
                            except asyncio.TimeoutError:
                                continue
                            except Exception as e:
                                logger.error("Lỗi gửi text: %s", e, exc_info=True)
                                break

                    async def heartbeat_loop():
                        while active():
                            try:
                                await asyncio.sleep(5)
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=SILENT_CHUNK_16K,
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                )
                            except Exception as e:
                                logger.error("Heartbeat lỗi: %s", e)
                                break

                    send_audio_task = asyncio.create_task(send_audio_loop())
                    send_text_task = asyncio.create_task(send_text_loop())
                    heartbeat_task = asyncio.create_task(heartbeat_loop())

                    try:
                        async for response in session.receive():
                            if not active():
                                break

                            if response.session_resumption_update:
                                update = response.session_resumption_update
                                if update.new_handle:
                                    session_handle = update.new_handle

                            self._log_response(response)

                            if (
                                response.server_content
                                and response.server_content.interrupted
                                and audio_interrupt_callback
                            ):
                                logger.info("Gemini: user interrupted")
                                if inspect.iscoroutinefunction(
                                    audio_interrupt_callback
                                ):
                                    await audio_interrupt_callback()
                                else:
                                    audio_interrupt_callback()

                            if (
                                response.server_content
                                and response.server_content.turn_complete
                            ):
                                logger.info("Gemini: turn complete — chờ user nói tiếp")

                            audio_data = self._extract_audio(response)
                            if audio_data and audio_output_callback:
                                if inspect.iscoroutinefunction(audio_output_callback):
                                    await audio_output_callback(audio_data)
                                else:
                                    audio_output_callback(audio_data)

                    except Exception as e:
                        logger.error(
                            "Lỗi nhận response: %s\n%s", e, traceback.format_exc()
                        )
                    finally:
                        send_audio_task.cancel()
                        send_text_task.cancel()
                        heartbeat_task.cancel()
                        for t in (send_audio_task, send_text_task, heartbeat_task):
                            try:
                                await t
                            except asyncio.CancelledError:
                                pass

            except Exception as e:
                logger.error(
                    "Gemini Live lỗi (model=%s): %s\n%s",
                    self.model,
                    e,
                    traceback.format_exc(),
                )

            if not active():
                break

            logger.warning("Gemini receive stream đóng — reconnect sau 1s...")
            await asyncio.sleep(1)

        logger.info("Gemini session kết thúc (audio_sent=%d)", audio_sent_count)

    @staticmethod
    def _log_response(response):
        if not response.server_content:
            return
        sc = response.server_content
        if sc.input_transcription and sc.input_transcription.text:
            logger.info("User nói: %s", sc.input_transcription.text)
        if sc.output_transcription and sc.output_transcription.text:
            logger.info("Gemini nói: %s", sc.output_transcription.text)

    @staticmethod
    def _extract_audio(response) -> bytes | None:
        if hasattr(response, "data") and response.data:
            return response.data

        if response.server_content and response.server_content.model_turn:
            for part in response.server_content.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
        return None
