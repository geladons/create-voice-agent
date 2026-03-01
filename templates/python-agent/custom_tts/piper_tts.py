"""
Piper TTS Plugin for LiveKit Agents.
Based on the community plugin by nay-cat/LiveKit-PiperTTS-Plugin.
Uses the piper-tts Python package for fully offline text-to-speech.
"""

import json
import numpy as np
import asyncio
import logging
import onnxruntime as ort
import threading
import uuid

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from piper import PiperVoice, SynthesisConfig
from piper.config import PiperConfig

logger = logging.getLogger("piper-tts")


class PiperTTSPlugin(tts.TTS):
    def __init__(
        self,
        model,
        speed=1.0,
        volume=1.0,
        noise_scale=0.667,
        noise_w=0.8,
        use_cuda=False,
        ort_intra_threads=1,
        ort_inter_threads=1,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=22050,
            num_channels=1,
        )
        self._tts_model_path = model
        self.speed = speed
        self.volume = volume
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.use_cuda = use_cuda
        self.ort_intra_threads = max(int(ort_intra_threads), 1)
        self.ort_inter_threads = max(int(ort_inter_threads), 1)
        self._voice_lock = threading.RLock()
        self._voice = None
        self._load_voice()

    def _load_voice(self, model_path=None):
        target_model = str(model_path or self._tts_model_path)
        logger.info("Loading Piper voice model: %s", target_model)
        config_path = f"{target_model}.json"
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        providers = (
            [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
            if self.use_cuda
            else ["CPUExecutionProvider"]
        )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.ort_intra_threads
        session_options.inter_op_num_threads = self.ort_inter_threads
        # Optimize for CPU inference
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Disable unnecessary optimizations that can cause overhead
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True

        session = ort.InferenceSession(
            target_model,
            sess_options=session_options,
            providers=providers,
        )

        with self._voice_lock:
            self._tts_model_path = target_model
            self._voice = PiperVoice(
                session=session,
                config=PiperConfig.from_dict(config_dict),
            )
        logger.info(
            "Piper voice model loaded successfully (intra_threads=%d, inter_threads=%d)",
            self.ort_intra_threads,
            self.ort_inter_threads,
        )

    def set_model(self, model_path):
        target_model = str(model_path)
        if target_model == self._tts_model_path:
            return
        logger.info("Switching Piper voice: %s -> %s", self._tts_model_path, target_model)
        self._load_voice(target_model)

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return PiperStream(self, text, conn_options)


class PiperStream(tts.ChunkedStream):
    def __init__(self, plugin, text, conn_options):
        super().__init__(tts=plugin, input_text=text, conn_options=conn_options)
        self.plugin = plugin

    async def _run(self, output_emitter, *args, **kwargs):
        try:
            output_emitter.initialize(
                request_id=f"piper-{uuid.uuid4().hex}",
                sample_rate=22050,
                num_channels=1,
                mime_type="audio/pcm",
                frame_size_ms=50,
            )

            config = SynthesisConfig(
                volume=self.plugin.volume,
                length_scale=self.plugin.speed,
                noise_scale=self.plugin.noise_scale,
                noise_w_scale=self.plugin.noise_w,
                normalize_audio=True,
            )

            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, self._synthesize_chunks, config)
            if not chunks:
                chunks = [np.zeros(22050, dtype=np.int16).tobytes()]

            for chunk in chunks:
                output_emitter.push(chunk)

            output_emitter.flush()

        except Exception as e:
            logger.error(f"Piper TTS synthesis failed: {e}")
            silence = np.zeros(22050, dtype=np.int16).tobytes()
            output_emitter.push(silence)
            output_emitter.flush()

    def _synthesize_chunks(self, config):
        chunks = []
        with self.plugin._voice_lock:
            voice = self.plugin._voice

        for chunk in voice.synthesize(self.input_text, syn_config=config):
            audio_data = chunk.audio_int16_bytes
            if chunk.sample_channels == 2:
                audio = np.frombuffer(audio_data, dtype=np.int16)
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                audio_data = audio.tobytes()
            chunks.append(audio_data)
        return chunks
