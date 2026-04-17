# main.py
#
# Single-file Modal pipeline:
#  - Upload local video/audio to Modal Volume
#  - Convert to mono 16kHz WAV (ffmpeg)
#  - Speaker diarization (pyannote)
#  - ASR with NVIDIA NeMo Parakeet TDT 0.6B v2 (English) + timestamps
#  - Align ASR segments to speakers by timestamp overlap
#  - Write /data/out.json and /data/out.txt in the volume
#
# Setup:
# 1) modal setup
# 2) Create Modal Secret "huggingface-secret" with env var HF_TOKEN
# 3) Accept required pyannote model terms on Hugging Face if prompted
#
# Run:
#   modal run main.py --local-input path/to/audio.wav
#   modal run main.py --local-input path/to/video.mp4
#
# Useful debug run (first N seconds only):
#   modal run main.py --local-input data/clip.wav --trim-secs 75

import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import modal

APP_NAME = "twin-speech-v0"
VOLUME_NAME = "twin-audio-data"
MODEL_CACHE_VOLUME_NAME = "twin-model-cache"

GPU = "A10G"
DATA_DIR = "/data"
MODEL_CACHE_DIR = "/model-cache"

HF_HOME = f"{MODEL_CACHE_DIR}/huggingface"
NEMO_CACHE_DIR = f"{MODEL_CACHE_DIR}/nemo"

IN_VOL_NAME = "input_media"
AUDIO_WAV_PATH = f"{DATA_DIR}/audio.wav"
AUDIO_TRIM_PATH = f"{DATA_DIR}/audio_trim.wav"
OUT_JSON_PATH = f"{DATA_DIR}/out.json"
OUT_TXT_PATH = f"{DATA_DIR}/out.txt"

# Diarization: Community-1 is reported by pyannote as better than 3.1 out of the box.
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-community-1"

# English-only Parakeet
NEMO_ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v2"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
model_cache_vol = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "numpy",
        "soundfile",
        "torch",
        "torchaudio",
        "pyannote.audio>=3.1.0",
        "nemo_toolkit[asr]",
        "omegaconf",
        "hydra-core",
        "sentencepiece",
    )
)

@app.function(volumes={DATA_DIR: vol})
def upload_to_volume(file_data: bytes, dest_name: str = IN_VOL_NAME) -> str:
    """Write file bytes into the Modal Volume; return absolute path in the volume."""
    dest = f"{DATA_DIR}/{dest_name}"
    with open(dest, "wb") as f:
        f.write(file_data)
    vol.commit()
    return dest


@app.function(volumes={DATA_DIR: vol})
def read_from_volume(vol_path: str) -> bytes:
    """Read a file from the Modal Volume and return its bytes."""
    with open(vol_path, "rb") as f:
        return f.read()


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={DATA_DIR: vol, MODEL_CACHE_DIR: model_cache_vol},
    env={
        "HF_HOME": HF_HOME,
        "NEMO_CACHE_DIR": NEMO_CACHE_DIR,
        # leave online access enabled so first-time model pulls work
        "HF_DATASETS_OFFLINE": "0",
    },
)
def run_pipeline(
    input_path_in_volume: str,
    wav_sr: int = 16000,
    trim_secs: int = 0,
    ) -> Dict[str, Any]:
    """
    input_path_in_volume: e.g. /data/input_media
    trim_secs: if > 0, only process the first N seconds
    """
    log("Pipeline started")
    log(f"  input:         {input_path_in_volume}")
    log(f"  trim_secs:     {trim_secs if trim_secs > 0 else 'none (full file)'}")
    log(f"  HF_HOME:       {os.environ.get('HF_HOME', 'not set')}")

    audio_wav = ensure_wav(input_path_in_volume, AUDIO_WAV_PATH, wav_sr)

    if trim_secs > 0:
        audio_wav = trim_wav(audio_wav, AUDIO_TRIM_PATH, trim_secs, wav_sr)

    diar = diarize_pyannote(
        audio_wav,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    diar_merged = merge_diar_segments(diar, max_gap_s=0.8)

    asr = transcribe_parakeet_nemo(audio_wav, model_name=NEMO_ASR_MODEL)
    merged = align_speakers_to_asr(diar, asr["segments"])

    payload = {
        "audio_wav": audio_wav,
        "diarization_segments": diar,
        "diarization_segments_merged": diar_merged,
        "asr_text": asr["text"],
        "asr_segments": asr["segments"],
        "speaker_labeled": merged,
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(OUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(render_speaker_transcript(merged))

    vol.commit()
    model_cache_vol.commit()

    log(f"Output written to {OUT_JSON_PATH}")
    log(f"Wrote transcript to {OUT_TXT_PATH}")

    return {
        "audio_wav": audio_wav,
        "out_json": OUT_JSON_PATH,
        "out_txt": OUT_TXT_PATH,
        "num_raw_diar_segments": len(diar),
        "num_merged_diar_segments": len(diar_merged),
        "num_asr_segments": len(asr["segments"]),
        "num_speaker_labeled_segments": len(merged),
        "preview_diar_merged": diar_merged[:20],
        "preview_speaker_labeled": merged[:30],
    }


# ----------------------------
# Helpers
# ----------------------------

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_wav(input_path: str, output_wav_path: str, sr: int) -> str:
    """Always convert input to fresh mono 16k WAV."""
    import subprocess

    log(f"[ensure_wav] Converting {input_path} -> {output_wav_path} at {sr}Hz mono")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        output_wav_path,
    ]
    subprocess.check_call(cmd)
    log("[ensure_wav] Done")
    return output_wav_path


def trim_wav(input_wav: str, output_wav: str, duration_secs: int, sr: int) -> str:
    """Trim and re-encode to stable mono PCM WAV."""
    import subprocess

    log(f"[trim_wav] Trimming first {duration_secs}s -> {output_wav}")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-t", str(duration_secs),
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        output_wav,
    ]
    subprocess.check_call(cmd)
    log("[trim_wav] Done")
    return output_wav


def diarize_pyannote(
    audio_wav_path: str,
    hf_token: Optional[str],
) -> List[Dict[str, Any]]:
    """Return diarization segments: [{'start':..,'end':..,'speaker':..}, ...]."""
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    log(f"[diarize] Loading {PYANNOTE_PIPELINE}")
    t0 = time.time()
    pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE, token=hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(torch.device(device))
    log(f"[diarize] Model loaded in {time.time() - t0:.1f}s (device={device})")

    infer_kwargs: Dict[str, Any] = {}

    log(f"[diarize] Running diarization on {audio_wav_path}")
    t1 = time.time()
    with ProgressHook() as hook:
        diarization = pipeline(audio_wav_path, hook=hook, **infer_kwargs)

    annotation = normalize_diarization_output(diarization)
    log(f"[diarize] Diarization done in {time.time() - t1:.1f}s")

    segments: List[Dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    segments.sort(key=lambda x: (x["start"], x["end"]))
    log(f"[diarize] {len(segments)} raw segments across {len({s['speaker'] for s in segments})} speakers")
    return segments


def normalize_diarization_output(diarization: Any) -> Any:
    """Handle pyannote output shape differences across versions."""
    if hasattr(diarization, "itertracks"):
        return diarization
    if isinstance(diarization, tuple) and len(diarization) > 0:
        first = diarization[0]
        if hasattr(first, "itertracks"):
            return first
    if hasattr(diarization, "speaker_diarization") and hasattr(diarization.speaker_diarization, "itertracks"):
        return diarization.speaker_diarization
    if hasattr(diarization, "annotation") and hasattr(diarization.annotation, "itertracks"):
        return diarization.annotation
    raise TypeError(f"Unsupported diarization output type: {type(diarization)}")


def merge_diar_segments(diar: List[Dict[str, Any]], max_gap_s: float = 0.8) -> List[Dict[str, Any]]:
    """Merge adjacent diarization segments for the same speaker when separated by a short gap."""
    if not diar:
        return []

    diar = sorted(diar, key=lambda x: (x["start"], x["end"]))
    out = [diar[0].copy()]

    for cur in diar[1:]:
        prev = out[-1]
        gap = cur["start"] - prev["end"]
        if cur["speaker"] == prev["speaker"] and gap <= max_gap_s:
            prev["end"] = max(prev["end"], cur["end"])
        else:
            out.append(cur.copy())

    return out


def transcribe_parakeet_nemo(audio_wav_path: str, model_name: str) -> Dict[str, Any]:
    """
    NeMo Parakeet ASR with timestamps.
    Returns:
      {
        "text": full_text,
        "segments": [{"start": s, "end": e, "text": segment_text}, ...]
      }
    """
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    import torch

    log(f"[asr] Loading {model_name}")
    t0 = time.time()
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model = asr_model.eval()
    log(f"[asr] Model loaded in {time.time() - t0:.1f}s")

    # Work around NeMo TDT CUDA-graphs decoder bug
    # Keep greedy batched decoding, but disable CUDA graph decoder explicitly.
    if hasattr(asr_model, "change_decoding_strategy"):
        decoding_cfg = OmegaConf.create(
            {
                "strategy": "greedy_batch",
                "fused_batch_size": 1,   # avoid the fused CUDA-graphs path
                "preserve_alignments": True,
                "greedy": {
                    "loop_labels": True,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
        asr_model.change_decoding_strategy(decoding_cfg)
        log("[asr] Applied decoding workaround: use_cuda_graph_decoder=False, fused_batch_size=1")

    log(f"[asr] Transcribing {audio_wav_path}")
    t1 = time.time()

    with torch.inference_mode():
        hyps = asr_model.transcribe(
            [audio_wav_path],
            timestamps=True,
            batch_size=1,
            verbose=False,
        )

    log(f"[asr] Transcription done in {time.time() - t1:.1f}s")

    hyp = hyps[0]
    full_text = getattr(hyp, "text", "")

    # NeMo logs indicate timestamps are exposed on timestep['word'/'segment'/'char']
    timestamp_obj = getattr(hyp, "timestamp", None)
    if timestamp_obj is None:
        timestamp_obj = getattr(hyp, "timestep", None)

    seg_ts = []
    if isinstance(timestamp_obj, dict):
        seg_ts = timestamp_obj.get("segment", []) or []
        if not seg_ts and "word" in timestamp_obj:
            # fallback: synthesize segment-like chunks from word timestamps
            for w in timestamp_obj["word"]:
                if "start" in w and "end" in w:
                    seg_ts.append(
                        {
                            "start": float(w["start"]),
                            "end": float(w["end"]),
                            "segment": str(w.get("word", "")).strip(),
                        }
                    )

    segments: List[Dict[str, Any]] = []
    for s in seg_ts:
        if "start" in s and "end" in s:
            segments.append(
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "text": str(s.get("segment", s.get("word", ""))).strip(),
                }
            )

    if not segments:
        segments = [{"start": 0.0, "end": 0.0, "text": full_text.strip()}]

    return {"text": full_text, "segments": segments}

def align_speakers_to_asr(
    diar: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign each ASR segment to the diarized speaker who overlaps its midpoint."""
    def speaker_for_time(t: float) -> str:
        for seg in diar:
            if seg["start"] <= t <= seg["end"]:
                return seg["speaker"]
        if not diar:
            return "UNKNOWN"
        best = min(diar, key=lambda s: min(abs(t - s["start"]), abs(t - s["end"])))
        return best["speaker"]

    merged: List[Dict[str, Any]] = []
    for seg in asr_segments:
        s, e = float(seg["start"]), float(seg["end"])
        mid = (s + e) / 2.0 if e > s else s
        merged.append(
            {
                "start": s,
                "end": e,
                "speaker": speaker_for_time(mid),
                "text": str(seg.get("text", "")).strip(),
            }
        )

    merged = coalesce_adjacent(merged, max_gap_s=0.6)
    return merged


def coalesce_adjacent(items: List[Dict[str, Any]], max_gap_s: float = 0.6) -> List[Dict[str, Any]]:
    """Merge adjacent ASR segments if same speaker and gap is small."""
    if not items:
        return []

    out = [items[0].copy()]
    for cur in items[1:]:
        prev = out[-1]
        gap = cur["start"] - prev["end"]
        if cur["speaker"] == prev["speaker"] and gap <= max_gap_s:
            prev["end"] = max(prev["end"], cur["end"])
            prev["text"] = (prev["text"].rstrip() + " " + cur["text"].lstrip()).strip()
        else:
            out.append(cur.copy())
    return out


def render_speaker_transcript(merged: List[Dict[str, Any]]) -> str:
    lines = []
    for seg in merged:
        lines.append(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['speaker']}: {seg['text']}")
    return "\n".join(lines) + "\n"


# ----------------------------
# Local entrypoint
# ----------------------------

@app.local_entrypoint()
def main(
    local_input: str = "",
    vol_input: str = "",
    trim_secs: int = 0,
):
    """
    Examples:
      modal run main.py --local-input path/to/audio.wav
      modal run main.py --local-input path/to/audio.wav --trim-secs 75
      modal run main.py --vol-input input_media
    """
    if vol_input:
        in_vol_path = vol_input if vol_input.startswith("/") else f"{DATA_DIR}/{vol_input}"
        print(f"Using existing volume file: {in_vol_path}")
    else:
        p = pathlib.Path(local_input)
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {local_input}")
        file_data = p.read_bytes()
        in_vol_path = upload_to_volume.remote(file_data, dest_name=IN_VOL_NAME)
        print(f"Uploaded to volume: {in_vol_path}")

    if trim_secs > 0:
        print(f"Trimming to first {trim_secs}s for this run")

    result = run_pipeline.remote(
        in_vol_path,
        trim_secs=trim_secs
    )

    print("\n=== Pipeline complete ===")
    print(json.dumps(result, indent=2))

    print("\n=== Preview: merged diarization ===")
    for row in result["preview_diar_merged"]:
        print(f"[{row['start']:.2f}–{row['end']:.2f}] {row['speaker']}")

    print("\n=== Preview: first labeled transcript chunks ===")
    for row in result["preview_speaker_labeled"]:
        print(f"[{row['start']:.2f}–{row['end']:.2f}] {row['speaker']}: {row['text']}")

    print("\nOutputs written in the volume:")
    print(f"  {result['out_json']}")
    print(f"  {result['out_txt']}")

    local_out_dir = pathlib.Path("data/output")
    local_out_dir.mkdir(parents=True, exist_ok=True)

    for vol_path in (result["out_json"], result["out_txt"]):
        data = read_from_volume.remote(vol_path)
        local_path = local_out_dir / pathlib.Path(vol_path).name
        local_path.write_bytes(data)
        print(f"Downloaded {vol_path} -> {local_path}")

    print("Done Running Pipeline")

