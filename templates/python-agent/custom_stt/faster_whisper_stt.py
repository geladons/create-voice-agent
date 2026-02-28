"""
Faster Whisper STT Plugin for LiveKit Agents.
Uses the faster-whisper Python package for fully offline speech-to-text.
"""

import asyncio
import io
import logging
import os
import wave

import numpy as np
from faster_whisper import WhisperModel

from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("faster-whisper-stt")


class FasterWhisperSTT(stt.STT):
    def __init__(
        self,
        model="base",
        language="en",
        device="cpu",
        compute_type="int8",
        download_root=None,
        local_files_only=None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self.model_size = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root or os.getenv(
            "FASTER_WHISPER_MODEL_CACHE_DIR",
            "/app/models/faster-whisper",
        )
        if local_files_only is None:
            local_files_only = os.getenv("FASTER_WHISPER_LOCAL_FILES_ONLY", "0")
        self.local_files_only = str(local_files_only).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._model = None
        self._load_model()

    def _load_model(self):
        logger.info(
            "Loading Whisper model: %s (device=%s, cache_dir=%s, local_files_only=%s)",
            self.model_size,
            self.device,
            self.download_root,
            self.local_files_only,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
            local_files_only=self.local_files_only,
        )
        logger.info("Whisper model loaded successfully")

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe, buffer, language or self.language
        )

    def _stream_impl(self, *, language=None, conn_options=None):
        raise NotImplementedError("FasterWhisperSTT does not support streaming")

    def _transcribe(self, buffer, language):
        # Convert LiveKit AudioFrame to numpy array
        if hasattr(buffer, "data"):
            audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = buffer.sample_rate if hasattr(buffer, "sample_rate") else 16000
        elif hasattr(buffer, "frame"):
            frame = buffer.frame
            audio_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = frame.sample_rate if hasattr(frame, "sample_rate") else 16000
        else:
            audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = 16000

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = resample(audio_data, num_samples).astype(np.float32)

        segments, info = self._model.transcribe(
            audio_data,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(segment.text for segment in segments).strip()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=language or info.language,
                    confidence=1.0,
                )
            ],
        )
