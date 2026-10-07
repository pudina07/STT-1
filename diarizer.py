"""
diarizer.py  --  OPTIONAL stage between STT and refinement: "who said it?"

Model : pyannote/speaker-diarization-3.1 (Hugging Face; gated - see README steps below)
Input : the original audio file + the TranscriptionResult from transcriber.py
Output: the same TranscriptionResult, but every segment now carries `speaker` ("Speaker 1", ...)

Why it matters
--------------
Action items are almost always phrased relative to the speaker: "I will send the report by Friday".
Without knowing WHO said "I", the extractor can only answer "Unspecified". With speaker labels in
the transcript the extractor can ground the owner of every first-person commitment:
"Speaker 2: I'll send the report" -> owner = whoever Speaker 2 is (named later by the extractor
from the transcript itself, e.g. when the chair says "Thank you, Councillor Miller").

How it works
------------
1. diarize()            : decode audio with ffmpeg -> pyannote pipeline -> [(start, end, speaker)].
2. apply_diarization()  : align every Whisper WORD to the speaker who talks most during it, smooth
                          one-word flickers, and split Whisper segments where the speaker changes.
                          Speakers are renamed by order of first appearance (Speaker 1, 2, ...).
Timestamps are used for the alignment only; they never appear in any output.

One-time setup
--------------
1. pip install pyannote.audio torch            (Python >= 3.10; GPU optional but much faster)
2. Create a Hugging Face READ token: https://huggingface.co/settings/tokens
3. Open both pages while logged in and click "Agree and access repository":
      https://huggingface.co/pyannote/speaker-diarization-3.1
      https://huggingface.co/pyannote/segmentation-3.0
4. Put  HF_TOKEN=hf_xxx  in your .env (or paste it in the app sidebar).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from transcriber import Segment, TranscriptionResult

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
SAMPLE_RATE = 16000

ProgressCB = Optional[Callable[[float, str], None]]


class DiarizationError(Exception):
    """Raised with a message that is safe to show directly to the end user."""


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    segments: list[DiarSegment]
    model: str
    device: str = "cpu"
    warnings: list[str] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for s in sorted(self.segments, key=lambda x: x.start):
            if s.speaker not in seen:
                seen.append(s.speaker)
        return seen


# --------------------------------------------------------------------------------------
# Availability / audio loading / pipeline
# --------------------------------------------------------------------------------------
def diarization_available() -> tuple[bool, str]:
    """(ok, reason). Cheap check that does not load any model."""
    try:
        import pyannote.audio  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:  # ImportError, or a broken torch install
        return False, f"pyannote.audio / torch are not installed ({type(e).__name__}). Run: pip install pyannote.audio torch"
    return True, ""


def _load_audio(path: Path):
    """Decode ANY audio/video file to a mono 16 kHz float32 torch tensor (1, T) using ffmpeg,
    so we never depend on torchaudio/torchcodec being able to read the container."""
    import numpy as np
    import torch

    if not shutil.which("ffmpeg"):
        raise DiarizationError("ffmpeg not found - install it first (it is also needed for transcription).")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise DiarizationError("Could not decode the audio for speaker analysis.")
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return {"waveform": torch.from_numpy(arr).unsqueeze(0), "sample_rate": SAMPLE_RATE}


_PIPELINE_CACHE: dict[tuple[str, str], object] = {}


def _load_pipeline(model: str, token: str):
    """Load (and cache) the pyannote pipeline. pyannote 3.x calls the kwarg `use_auth_token`,
    4.x calls it `token`, so we try both."""
    key = (model, token[-6:])
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise DiarizationError(
            f"pyannote.audio is not installed or failed to import ({type(e).__name__}: {e}). "
            "Run: pip install pyannote.audio torch"
        )
    pipeline = None
    err: Optional[Exception] = None
    for kw in ("token", "use_auth_token"):
        try:
            pipeline = Pipeline.from_pretrained(model, **{kw: token})
            err = None
            break
        except TypeError as e:          # wrong kwarg name for this pyannote version
            err = e
            continue
        except Exception as e:
            err = e
            break
    if pipeline is None:
        text = str(err or "")
        low = text.lower()
        if any(k in low for k in ("401", "403", "gated", "access", "authorized", "token", "restricted")) or err is None:
            raise DiarizationError(
                "Hugging Face refused access to the diarization model. Check that (1) HF_TOKEN is a valid READ token and "
                "(2) you clicked 'Agree and access repository' on BOTH https://huggingface.co/pyannote/speaker-diarization-3.1 "
                "and https://huggingface.co/pyannote/segmentation-3.0 with the same account."
            )
        raise DiarizationError(f"Could not load the diarization model: {type(err).__name__}: {text[:300]}")
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    _PIPELINE_CACHE[key] = pipeline
    return pipeline


def _device_name() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------------------
# Step 1: run pyannote
# --------------------------------------------------------------------------------------
def diarize(
    audio_path: str | Path,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    model: Optional[str] = None,
    progress_cb: ProgressCB = None,
) -> DiarizationResult:
    report = progress_cb or (lambda f, m: None)
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise DiarizationError(
            "HF_TOKEN is missing. Add your Hugging Face read token to .env (or the sidebar) to enable speaker labels."
        )
    model = model or DEFAULT_DIARIZATION_MODEL
    path = Path(audio_path)
    if not path.exists():
        raise DiarizationError("Audio file not found for speaker analysis.")

    report(0.05, "Loading speaker model...")
    pipeline = _load_pipeline(model, token)
    report(0.2, "Decoding audio...")
    audio = _load_audio(path)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)

    report(0.3, "Finding who speaks when (this is the slow part on CPU)...")
    try:
        out = pipeline(audio, **kwargs)
    except Exception as e:
        raise DiarizationError(f"Speaker analysis failed: {type(e).__name__}: {str(e)[:300]}")

    annotation = getattr(out, "speaker_diarization", out)   # 4.x returns an object, 3.x an Annotation
    segs: list[DiarSegment] = []
    try:
        for turn, _, label in annotation.itertracks(yield_label=True):
            if turn.end - turn.start >= 0.05:
                segs.append(DiarSegment(float(turn.start), float(turn.end), str(label)))
    except Exception as e:
        raise DiarizationError(f"Unexpected diarization output ({type(e).__name__}).")
    if not segs:
        raise DiarizationError("No speech turns were found by the speaker model.")
    report(1.0, "Speaker analysis complete.")
    return DiarizationResult(sorted(segs, key=lambda s: s.start), model=model, device=_device_name())


# --------------------------------------------------------------------------------------
# Step 2: merge speakers into the transcript
# --------------------------------------------------------------------------------------
def _speaker_for_span(a: float, b: float, starts: list[float], segs: list[DiarSegment],
                      max_gap: float = 1.5) -> Optional[str]:
    """Speaker with the largest overlap with [a, b]; else the nearest turn within max_gap seconds."""
    overlap: dict[str, float] = {}
    i = max(0, bisect_left(starts, a) - 8)
    # diarization turns can overlap, so scan a small window rather than a single index
    while i < len(segs) and segs[i].start <= b:
        s = segs[i]
        ov = min(b, s.end) - max(a, s.start)
        if ov > 0:
            overlap[s.speaker] = overlap.get(s.speaker, 0.0) + ov
        i += 1
    if overlap:
        return max(overlap.items(), key=lambda kv: kv[1])[0]
    mid = (a + b) / 2
    best, best_d = None, max_gap
    j = max(0, bisect_left(starts, mid) - 8)
    while j < len(segs) and segs[j].start - mid < max_gap:
        s = segs[j]
        d = 0.0 if s.start <= mid <= s.end else min(abs(s.start - mid), abs(s.end - mid))
        if d < best_d:
            best, best_d = s.speaker, d
        j += 1
    return best


def _smooth(labels: list[Optional[str]], spans: list[tuple[float, float]], max_words: int = 2,
            max_dur: float = 0.9) -> list[Optional[str]]:
    """Remove 1-2 word speaker 'flickers' sandwiched between the same speaker."""
    out = list(labels)
    # fill unknowns from neighbours first
    for i, l in enumerate(out):
        if l is None:
            prev = next((out[k] for k in range(i - 1, -1, -1) if out[k]), None)
            nxt = next((out[k] for k in range(i + 1, len(out)) if out[k]), None)
            out[i] = prev or nxt
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[i]:
            j += 1
        n = j - i + 1
        dur = spans[j][1] - spans[i][0]
        if 0 < i and j < len(out) - 1 and n <= max_words and dur <= max_dur and out[i - 1] == out[j + 1] != out[i]:
            for k in range(i, j + 1):
                out[k] = out[i - 1]
        i = j + 1
    return out


def apply_diarization(result: TranscriptionResult, diar: DiarizationResult) -> TranscriptionResult:
    """Label every segment with a speaker (splitting segments at speaker changes when word timings
    are available). Returns the same TranscriptionResult object, modified in place."""
    segs = sorted(diar.segments, key=lambda s: s.start)
    starts = [s.start for s in segs]

    new_segments: list[Segment] = []
    for seg in result.segments:
        tokens = seg.text.split()
        words = seg.words
        if words and len(words) == len(tokens) and len(words) > 1:
            labels = [_speaker_for_span(w.start, w.end, starts, segs) for w in words]
            spans = [(w.start, w.end) for w in words]
            labels = _smooth(labels, spans)
            # split into runs of the same speaker
            run_start = 0
            for k in range(1, len(words) + 1):
                if k == len(words) or labels[k] != labels[run_start]:
                    new_segments.append(Segment(
                        start=words[run_start].start, end=words[k - 1].end,
                        text=" ".join(tokens[run_start:k]), speaker=labels[run_start],
                        words=words[run_start:k], avg_logprob=seg.avg_logprob,
                    ))
                    run_start = k
        else:
            seg.speaker = _speaker_for_span(seg.start, seg.end, starts, segs)
            new_segments.append(seg)

    # carry the previous speaker over segments that fell in a gap
    last = None
    for s in new_segments:
        if s.speaker is None:
            s.speaker = last
        last = s.speaker or last
    first = next((s.speaker for s in new_segments if s.speaker), None)
    for s in new_segments:
        if s.speaker is None:
            s.speaker = first

    # rename by order of first appearance
    names: dict[str, str] = {}
    for s in new_segments:
        if s.speaker and s.speaker not in names:
            names[s.speaker] = f"Speaker {len(names) + 1}"
    for s in new_segments:
        s.speaker = names.get(s.speaker) if s.speaker else None

    result.segments = new_segments
    if not result.has_word_timings:
        result.warnings.append(
            "Word-level timings were not available, so speakers were assigned per Whisper segment "
            "(a speaker change in the middle of a segment may be missed)."
        )
    return result


def speaker_stats(result: TranscriptionResult) -> dict[str, dict]:
    """words / seconds / share per speaker - for display in the UI."""
    stats: dict[str, dict] = {}
    for s in result.segments:
        if not s.speaker:
            continue
        d = stats.setdefault(s.speaker, {"words": 0, "seconds": 0.0})
        d["words"] += len(s.text.split())
        d["seconds"] += max(0.0, s.end - s.start)
    total = sum(d["words"] for d in stats.values()) or 1
    for d in stats.values():
        d["seconds"] = round(d["seconds"], 1)
        d["share"] = round(d["words"] / total, 3)
    return stats
