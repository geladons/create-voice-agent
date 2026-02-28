"""
Faster Whisper STT Plugin for LiveKit Agents.
Uses the faster-whisper Python package for fully offline speech-to-text.
"""

import asyncio
import io
import logging
import os
import threading
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
        device="auto",
        compute_type="int8",
        download_root=None,
        local_files_only=None,
        beam_size=None,
        best_of=None,
        vad_filter=None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self.model_size = model
        self.language = language
        self.device = os.getenv("FASTER_WHISPER_DEVICE", device)
        self.compute_type = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", compute_type)
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
        self.beam_size = self._coerce_int(
            beam_size if beam_size is not None else os.getenv("FASTER_WHISPER_BEAM_SIZE", "1"),
            default=1,
            minimum=1,
        )
        self.best_of = self._coerce_int(
            best_of if best_of is not None else os.getenv("FASTER_WHISPER_BEST_OF", "1"),
            default=1,
            minimum=1,
        )
        self.vad_filter = self._coerce_bool(
            vad_filter if vad_filter is not None else os.getenv("FASTER_WHISPER_VAD_FILTER", "1"),
            default=True,
        )
        self._model_lock = threading.RLock()
        self._model = None
        self._load_model()

    def _coerce_bool(self, value, default=True):
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _coerce_int(self, value, default=1, minimum=1):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if parsed < minimum:
            return default
        return parsed

    def _load_model(self):
        logger.info(
            (
                "Loading Whisper model: %s (device=%s, compute_type=%s, cache_dir=%s, "
                "local_files_only=%s, beam_size=%s, best_of=%s, vad_filter=%s)"
            ),
            self.model_size,
            self.device,
            self.compute_type,
            self.download_root,
            self.local_files_only,
            self.beam_size,
            self.best_of,
            self.vad_filter,
        )
        model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
            local_files_only=self.local_files_only,
        )
        with self._model_lock:
            self._model = model
        logger.info("Whisper model loaded successfully")

    def set_language(self, language):
        normalized = str(language or "").strip().lower()
        if normalized:
            self.language = normalized

    def set_model(self, model):
        requested = str(model or "").strip()
        if not requested or requested == self.model_size:
            return False
        self.model_size = requested
        self._load_model()
        return True

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        loop = asyncio.get_running_loop()
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

        with self._model_lock:
            model = self._model

        segments, info = model.transcribe(
            audio_data,
            language=language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
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
