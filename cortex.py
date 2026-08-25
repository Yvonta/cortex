import io
import re
import os
import sys
import wave
import types
import json
import hashlib
import asyncio
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import asynccontextmanager

import httpx
import requests
import numpy as np
import torch

import warnings

import tempfile
import whisper

# Suppress PyTorch SDP kernel deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.backends.cuda")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*sdp_kernel.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*sdp_kernel.*")

#if hasattr(torch.backends.cuda, "sdp_kernel"):
#    torch.backends.cuda.sdp_kernel = torch.nn.attention.sdpa_kernel

import ollama

from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile, Security, status, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from pydub import AudioSegment

# --- macOS / CPU WORKAROUND FOR HARDCODED CUDA CALLS ---
if not torch.cuda.is_available():
    torch.cuda.is_available = lambda: False


# --- WATERMARK PATCH START ---
class DummyWatermarker:
    def __init__(self, *args, **kwargs):
        pass
    def apply_watermark(self, wav, sample_rate=None):
        return wav

mock_perth = types.ModuleType("perth")
mock_perth.PerthImplicitWatermarker = DummyWatermarker
sys.modules["perth"] = mock_perth
# --- WATERMARK PATCH END ---

from chatterbox import ChatterboxTTS, ChatterboxMultilingualTTS

# Configuration
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
ID = 0
API_KEY_NAME = "X-API-Key"
EXPECTED_API_KEY = os.getenv("API_KEY")


app_state = {}

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key


def strip_all_hooks_and_analyzers(tts_model):
    """Safely traverses internal Chatterbox modules to disable attention output flags and clear hooks."""
    try:
        hook_count = 0
        # Chatterbox models store PyTorch submodules in attributes like t3, s1, ve, etc.
        submodules = []
        for attr_name in ["t3", "s1", "ve", "voice_encoder", "model", "patched_model"]:
            if hasattr(tts_model, attr_name):
                submodules.append(getattr(tts_model, attr_name))

        if not submodules and isinstance(tts_model, torch.nn.Module):
            submodules = [tts_model]

        for submod in submodules:
            if isinstance(submod, torch.nn.Module):
                for module in submod.modules():
                    if hasattr(module, "_forward_hooks"):
                        hook_count += len(module._forward_hooks)
                        module._forward_hooks.clear()
                    if hasattr(module, "_forward_pre_hooks"):
                        module._forward_pre_hooks.clear()

                    if hasattr(module, "config") and hasattr(module.config, "output_attentions"):
                        module.config.output_attentions = False

        print(f"[Fix] Successfully cleaned {hook_count} hooks across internal submodules.")
    except Exception as e:
        print(f"[Warning] Failed to apply optimization stripping: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading Chatterbox TTS model target device: {device}...")

    # Initialize a lock for serializing inference calls
    app_state["generate_lock"] = asyncio.Lock()

    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device)
        strip_all_hooks_and_analyzers(model)

        app_state["tts_model"] = model
        app_state["device"] = device
        print(f"Model loaded successfully on {device}.")
    except Exception as e:
        print(f"Failed to load model on {device}: {e}")
        if device == "mps":
            print("Retrying model initialization on CPU...")
            model = ChatterboxMultilingualTTS.from_pretrained("cpu")
            strip_all_hooks_and_analyzers(model)
            app_state["tts_model"] = model
            app_state["device"] = "cpu"
            print("Model loaded successfully on CPU fallback.")
        else:
            raise e

    yield
    app_state.clear()


app = FastAPI(title="Donkey API", lifespan=lifespan, dependencies=[Depends(verify_api_key)])


def split_text_into_chunks(text: str):
    """Splits text by sentence boundaries to enable instant audio streaming."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def pcm_to_encoded_bytes(pcm_array: np.ndarray, sample_rate: int, format: str) -> bytes:
    # Validate array dimensions and size
    if pcm_array is None or pcm_array.size == 0:
        return b""

    # Flatten if multidimensional (e.g. shape [1, N] -> [N])
    pcm_array = pcm_array.reshape(-1)

    if pcm_array.dtype in [np.float32, np.float64]:
        pcm_array = (pcm_array * 32767).clip(-32768, 32767).astype(np.int16)
    
    pcm_bytes = pcm_array.tobytes()

    if format.lower() == "wav":
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return wav_buffer.getvalue()

    try:
        audio_segment = AudioSegment(
            pcm_bytes,
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        buffer = io.BytesIO()
        audio_segment.export(buffer, format=format)
        return buffer.getvalue()
    except Exception as e:
        print(f"[Warning] Failed to export {format} using pydub: {e}. Falling back to WAV.")
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return wav_buffer.getvalue()


async def generate_audio_stream(text: str, format: str, voice_file_path: str = None, language_id: str = "en"):
    tts_model = app_state["tts_model"]
    gen_lock = app_state["generate_lock"]
    chunks = split_text_into_chunks(text)
    sample_rate = getattr(tts_model, "sample_rate", 24000)

    try:
        for chunk in chunks:
            clean_chunk = chunk.strip()
            if not clean_chunk:
                continue

            def synthesize():
                with torch.inference_mode():
                    kwargs = {"language_id": language_id} if language_id else {}
                    if voice_file_path:
                        kwargs["audio_prompt_path"] = voice_file_path
                    return tts_model.generate(clean_chunk, **kwargs)

            try:
                async with gen_lock:
                    audio_tensor = await asyncio.to_thread(synthesize)
            except Exception as e:
                print(f"[Warning] Error synthesizing chunk '{clean_chunk[:20]}...': {e}")
                continue

            if audio_tensor is None:
                continue

            if isinstance(audio_tensor, torch.Tensor):
                if audio_tensor.numel() == 0:
                    continue
                pcm_data = audio_tensor.detach().cpu().numpy()
            else:
                pcm_data = np.array(audio_tensor)

            if pcm_data is None or pcm_data.size == 0:
                continue

            chunk_bytes = pcm_to_encoded_bytes(pcm_data, sample_rate, format)
            if chunk_bytes:
                yield chunk_bytes

    finally:
        if voice_file_path and os.path.exists(voice_file_path):
            try:
                os.remove(voice_file_path)
            except OSError:
                pass
                
@app.post("/api/v1/tts/stream/upload")
@app.post("/api/v1/tts/stream")
async def stream_tts(
    text: str = Form(..., description="Text to synthesize"),
    format: str = Form("mp3", description="Audio format: mp3 or wav"),
    language_id: str = Form("en", description="Language code (e.g., nl, en)"),
    voice_file: UploadFile = File(None, description="Audio file for voice cloning")
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")

    temp_voice_path = None
    if voice_file:
        file_extension = os.path.splitext(voice_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await voice_file.read()
            temp_file.write(content)
            temp_voice_path = temp_file.name

    media_type = "audio/mpeg" if format == "mp3" else "audio/wav"

    return StreamingResponse(
        generate_audio_stream(text=text, format=format, voice_file_path=temp_voice_path, language_id=language_id),
        media_type=media_type,
        headers={
            "Content-Disposition": f"inline; filename=speech.{format}",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

"""hooks"""

def get_public_ip(api_url: str = "http://localhost/ip.php") -> str:
    req = urllib.request.Request(
        api_url, headers={"User-Agent": "Python-IP-Client/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("ip", "IP not found in response")

    except urllib.error.URLError as e:
        return f"Connection error: {e.reason}"
    except json.JSONDecodeError:
        return "Failed to decode JSON response"


def generate_api_secret():
    client_ip = get_public_ip("https://ultireal.com/appapi/v2/publicip.php")
    if not client_ip:
        return BASE_SECRET

    raw_string = f"{client_ip}:{BASE_SECRET}"
    return hashlib.sha256(raw_string.encode("utf-8")).hexdigest()


"""Ollama proxy"""

@app.post("/api/generate")
@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def proxy_ollama_stream(request: Request):
    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()
    target_url = f"{OLLAMA_BASE_URL}{request.url.path}"

    async def stream_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="application/json"
    )

""" Whisper """

# Load Whisper model on startup
# Options: 'tiny', 'base', 'small', 'medium', 'large'
MODEL_SIZE = "base"
model = whisper.load_model(MODEL_SIZE)

@app.get("/")
def read_root():
    return {"message": "Donkey API is running!"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = None):
    """
    Upload an audio file to transcribe it to text.
    Optional: Provide a language code (e.g., 'en', 'es', 'fr').
    """
    # Create a temporary file to store the uploaded audio
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        # Options for transcription
        options = {}
        if language:
            options["language"] = language

        # Run transcription using Whisper
        result = model.transcribe(temp_file_path, **options)

        return {
            "filename": file.filename,
            "detected_language": result.get("language"),
            "text": result.get("text").strip(),
            "segments": result.get("segments")  # Includes detailed timestamps
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary audio file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



if __name__ == "__main__":
    print("GPU Receiver online. Listening on http://0.0.0.0:8000...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
