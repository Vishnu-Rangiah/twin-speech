# Twin Speech

Post-meeting speech coach. Ingests meeting audio, identifies high-impact
coaching moments, and generates actionable feedback with concrete rewrites --
linked to the exact timestamp in the recording.

Nobody tells you when you ramble, mispronounce a key term, or bury your ask.
Twin Speech does, privately, after the meeting.

## What It Does

- **Transcription** -- speaker-attributed transcript with segment-level
  timestamps, using NVIDIA NeMo Parakeet TDT (ASR) and pyannote
  (speaker diarization)
- **Coaching cards** -- "you said X, here's why it matters, here's how to say
  it better" with 2-4 rewrite variants and a practice drill
- **Pronunciation analysis** -- phoneme-level detection of mispronounced or
  confused words with mouth cues and minimal pairs
- **Meeting summary** -- decisions, action items, per-speaker themes, and a
  "next meeting focus" skill

## Pipeline

The core pipeline lives in `main.py` and runs on [Modal](https://modal.com):

1. Upload local audio/video to a Modal Volume
2. Convert to mono 16 kHz WAV (`ffmpeg`)
3. Speaker diarization (`pyannote/speaker-diarization-community-1`)
4. ASR with `nvidia/parakeet-tdt-0.6b-v2` (English, with timestamps)
5. Align ASR segments to speakers by timestamp overlap
6. Write `out.json` and `out.txt` back to the volume and download locally

## Setup

1. `modal setup`
2. Create a Modal Secret named `huggingface-secret` with `HF_TOKEN`
3. Accept the pyannote model terms on Hugging Face if prompted

## Run

```sh
modal run main.py --local-input path/to/audio.wav
modal run main.py --local-input path/to/video.mp4

# Debug: only process the first N seconds
modal run main.py --local-input path/to/audio.wav --trim-secs 75
```

Outputs are downloaded to `data/output/`.

## Local Development

Remote dependencies (pyannote, NeMo, torch, etc.) are installed inside the
Modal image and do not need to be installed locally. The only local
requirement is `modal` itself, declared in `pyproject.toml`.

```sh
uv sync
```

## License

MIT License - see [LICENSE](LICENSE) for details.
