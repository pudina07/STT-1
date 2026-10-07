"""
transcriber.py  --  STAGE 1: speech-to-text

Model : Whisper large-v3-turbo, served by Groq (model id: "whisper-large-v3-turbo")
Input : any meeting recording (mp3 / wav / m4a / flac / ogg / webm / mp4 ...)
Output: TranscriptionResult -> segments (+ word timings) and a clean, TIMESTAMP-FREE transcript

What this module does
---------------------
1. Validates the file (exists, supported extension, non-empty, decodable, has audio,
   not too short / too long) and raises a human-readable TranscriptionError otherwise.
2. Uses ffmpeg to normalise audio to 16 kHz mono FLAC (lossless + small).
3. Splits long recordings into ~10 min chunks with a few seconds of overlap; every word /
   segment is "owned" by exactly one chunk (by its midpoint), so overlap never duplicates text.
   Each chunk is size-checked before upload: if the lossless FLAC is above Groq's 25 MB request
   limit it is re-encoded (64 kbps MP3, then 32 kbps), and as a last resort split in half, so any
   length / bitrate of input (e.g. a 25-min 260 MB WAV) is accepted.
4. Calls Groq with verbose_json and asks for BOTH word- and segment-level timings.
   Timings are used INTERNALLY only (to align speakers from the diarizer); they are never
   printed in the transcript. Falls back to segment-only if word timings are refused.
5. Drops obvious Whisper silence-hallucinations (very low confidence / endless repeats).
   Keeps Whisper's per-segment avg_logprob -> an "ASR clarity" score per paragraph, used later by
   confidence.py to tell the user when a decision rests on a poorly-heard passage.
6. build_turns(): groups segments into readable paragraphs / speaker turns - the unit that
   the refiner and extractor work on.

Requires: ffmpeg + ffprobe on PATH, `pip install groq python-dotenv`, env var GROQ_API_KEY.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".webm",
    ".mp4", ".mpeg", ".mpga", ".aac", ".wma", ".mkv", ".mov",
}
MIN_DURATION_S = 1.0
MAX_DURATION_S = 2 * 60 * 60          # 2 hours cap
CHUNK_SECONDS = int(os.getenv("STT_CHUNK_SECONDS", "600"))   # 10 minutes
OVERLAP_SECONDS = 3.0
MAX_UPLOAD_BYTES = int(float(os.getenv("STT_MAX_UPLOAD_MB", "24")) * 1024 * 1024)  # Groq free tier: 25 MB/request
MAX_ATTEMPTS = 5
MIN_SPLIT_SECONDS = 30.0              # never split a chunk below this when shrinking for upload

ProgressCB = Optional[Callable[[float, str], None]]


class TranscriptionError(Exception):
    """Raised with a message that is safe to show directly to the end user."""


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------
@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None          # "Speaker 1", ... (filled in by diarizer.apply_diarization)
    words: list[Word] = field(default_factory=list)
    avg_logprob: Optional[float] = None    # Whisper's mean token log-probability for this segment


@dataclass
class Turn:
    """A readable paragraph: one speaker's continuous talk (or a pause-delimited paragraph when
    diarization is off). Timings are kept only for internal alignment and never displayed."""
    speaker: Optional[str]
    text: str
    start: float = 0.0
    end: float = 0.0
    asr_conf: Optional[float] = None       # 0..1 Whisper clarity (exp of duration-weighted avg_logprob)


def logprob_to_conf(lp: Optional[float]) -> Optional[float]:
    if lp is None:
        return None
    return round(max(0.0, min(1.0, math.exp(lp))), 3)


def fmt_ts(seconds: Optional[float]) -> str:
    """12.3 -> '00:12', 3725 -> '1:02:05'."""
    if seconds is None:
        return "--:--"
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


_SENT_END = re.compile(r"[.!?][\"')\]]*$")


def build_turns(segments: list[Segment], max_chars: int = 900, soft_chars: int = 600,
                pause_s: float = 2.5) -> list[Turn]:
    """Group Whisper fragments into paragraphs.
    * a new turn starts when the speaker changes (if known) or after a long pause;
    * long turns are split at a sentence end once they pass `soft_chars` (hard cap `max_chars`)."""
    turns: list[Turn] = []
    cur: Optional[Turn] = None
    acc: list[float] = [0.0, 0.0]          # [sum(lp * dur), sum(dur)] for the current turn

    def _close():
        if cur is not None:
            cur.asr_conf = logprob_to_conf(acc[0] / acc[1]) if acc[1] > 0 else None
            turns.append(cur)

    for seg in segments:
        t = seg.text.strip()
        if not t:
            continue
        new = (
            cur is None
            or seg.speaker != cur.speaker
            or seg.start - cur.end > pause_s
            or len(cur.text) + 1 + len(t) > max_chars
            or (len(cur.text) >= soft_chars and _SENT_END.search(cur.text) is not None)
        )
        if new:
            _close()
            cur = Turn(seg.speaker, t, seg.start, seg.end)
            acc = [0.0, 0.0]
        else:
            cur.text += " " + t
            cur.end = seg.end
        if seg.avg_logprob is not None:
            dur = max(0.2, seg.end - seg.start)
            acc[0] += seg.avg_logprob * dur
            acc[1] += dur
    _close()
    return turns


def render_turns(turns: list[Turn], timestamps: bool = False) -> str:
    """Plain readable text: 'Speaker 1: ...' blocks when speakers are known, otherwise paragraphs.
    timestamps=True prefixes each block with its start time, e.g. '[03:12] Speaker 2: ...'."""
    blocks = []
    for t in turns:
        b = f"{t.speaker}: {t.text}" if t.speaker else t.text
        if timestamps:
            b = f"[{fmt_ts(getattr(t, 'start', None))}] {b}"
        blocks.append(b)
    return "\n\n".join(blocks)


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    duration_s: float
    model: str
    language: str = "en"
    chunks: int = 1
    has_word_timings: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def has_speakers(self) -> bool:
        return any(s.speaker for s in self.segments)

    def turns(self) -> list[Turn]:
        return build_turns(self.segments)

    def text(self, timestamps: bool = False) -> str:
        """Transcript (speaker-labelled when diarization has been applied); timestamps optional."""
        return render_turns(self.turns(), timestamps=timestamps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------------------
def _require_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise TranscriptionError(
            "ffmpeg/ffprobe not found on this machine. Install it first "
            "(Windows: `winget install ffmpeg`; macOS: `brew install ffmpeg`; "
            "Ubuntu/Debian: `sudo apt install ffmpeg`) and restart the app."
        )


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TranscriptionError("Audio conversion timed out. The file may be too large or damaged.")


def validate_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise TranscriptionError("The uploaded file could not be found.")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise TranscriptionError(
            f"Unsupported file type '{p.suffix or 'unknown'}'. Supported formats: "
            + ", ".join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))
            + "."
        )
    if p.stat().st_size == 0:
        raise TranscriptionError("The file is empty (0 bytes). Please upload a valid recording.")
    return p


def probe_audio(path: Path) -> Optional[float]:
    """Return duration in seconds (None if the container doesn't report it).
    Raises TranscriptionError when the file is unreadable or has no audio stream."""
    proc = _run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path)],
        timeout=120,
    )
    if proc.returncode != 0:
        raise TranscriptionError(
            "The file could not be read as audio. It may be corrupted or not a real media file."
        )
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        raise TranscriptionError("The file could not be analysed as audio (corrupted?).")
    if not info.get("streams"):
        raise TranscriptionError("The file contains no audio track.")
    try:
        return float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        return None


def _to_flac(src: Path, dst: Path, start: Optional[float] = None, length: Optional[float] = None) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dst)]
    proc = _run(cmd)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise TranscriptionError(
            "Audio conversion failed - the recording appears to be damaged or uses an unsupported codec."
        )


def _encode(src: Path, dst: Path, start: Optional[float], length: Optional[float], codec: list[str]) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", *codec, str(dst)]
    proc = _run(cmd)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def encode_chunk_for_upload(src: Path, tmpdir: Path, name: str, start: float, length: float,
                            limit: Optional[int] = None) -> Optional[Path]:
    """Encode one chunk so that it fits under the STT upload limit.
    Tries lossless FLAC first, then compressed MP3 at 64 / 32 kbps (speech stays fully intelligible).
    Returns None when even 32 kbps is too big (caller then splits the chunk in half)."""
    limit = limit or MAX_UPLOAD_BYTES
    flac = tmpdir / f"{name}.flac"
    _to_flac(src, flac, start=start, length=length)
    if flac.stat().st_size <= limit:
        return flac
    for kbps in (64, 32):
        mp3 = tmpdir / f"{name}_{kbps}k.mp3"
        if _encode(src, mp3, start, length, ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"]) \
                and mp3.stat().st_size <= limit:
            logger.info("Chunk %s re-encoded to %d kbps MP3 to fit the upload limit", name, kbps)
            return mp3
        ogg = tmpdir / f"{name}_{kbps}k.ogg"        # ffmpeg builds without lame
        if _encode(src, ogg, start, length, ["-c:a", "libopus", "-b:a", f"{kbps}k"]) \
                and ogg.stat().st_size <= limit:
            return ogg
    return None


def make_playback_audio(src: str | Path, max_seconds: Optional[float] = None) -> Optional[tuple[bytes, str]]:
    """Small mono copy of the recording for the in-app 'jump to timestamp' player
    (a 25 min meeting -> ~6 MB instead of a 260 MB WAV). Returns (bytes, mime) or None."""
    if not (shutil.which("ffmpeg")):
        return None
    src = Path(src)
    with tempfile.TemporaryDirectory(prefix="play_") as tmp:
        for ext, codec, mime in (("mp3", ["-c:a", "libmp3lame", "-b:a", "48k"], "audio/mpeg"),
                                 ("ogg", ["-c:a", "libopus", "-b:a", "32k"], "audio/ogg"),
                                 ("m4a", ["-c:a", "aac", "-b:a", "48k"], "audio/mp4")):
            dst = Path(tmp) / f"playback.{ext}"
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "22050"]
            if max_seconds:
                cmd += ["-t", f"{max_seconds:.0f}"]
            try:
                proc = subprocess.run(cmd + codec + [str(dst)], capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                return None
            if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return dst.read_bytes(), mime
    return None


def plan_chunks(total: float, chunk_s: float = CHUNK_SECONDS, overlap_s: float = OVERLAP_SECONDS):
    """Return [(cut_start, cut_end, own_start, own_end), ...].
    Audio is cut with overlap, but each chunk only 'owns' [own_start, own_end)."""
    if total <= chunk_s * 1.2:
        return [(0.0, total, 0.0, total)]
    n = math.ceil(total / chunk_s)
    if n > 1 and total - (n - 1) * chunk_s < 10:      # avoid a tiny trailing chunk
        n -= 1
    plans = []
    for i in range(n):
        own_s = i * chunk_s
        own_e = total if i == n - 1 else (i + 1) * chunk_s
        plans.append((max(0.0, own_s - overlap_s), min(total, own_e + overlap_s), own_s, own_e))
    return plans


# --------------------------------------------------------------------------------------
# Groq calls
# --------------------------------------------------------------------------------------
def _make_client(api_key: Optional[str]):
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise TranscriptionError(
            "GROQ_API_KEY is missing. Add it to your .env file (or paste it in the sidebar)."
        )
    from groq import Groq
    return Groq(api_key=key, timeout=180.0, max_retries=0)  # we do our own retry loop


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _call_groq(client, path: Path, model: str, prompt: Optional[str]):
    """Returns (response, got_word_timings). Tries word+segment timings first, then segment-only."""
    import groq

    base: dict[str, Any] = dict(model=model, response_format="verbose_json", language="en", temperature=0.0)
    if prompt:
        base["prompt"] = prompt[:800]   # Whisper prompt window is ~224 tokens

    granularities = [["word", "segment"], ["segment"]]
    gi = 0
    last_err: Optional[Exception] = None
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            with open(path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    file=(path.name, f.read()), timestamp_granularities=granularities[gi], **base)
            return resp, granularities[gi] == ["word", "segment"]
        except groq.AuthenticationError:
            raise TranscriptionError("Groq rejected the API key. Check GROQ_API_KEY.")
        except groq.BadRequestError as e:
            msg = str(getattr(e, "message", e))
            if gi + 1 < len(granularities) and ("granular" in msg.lower() or "timestamp" in msg.lower()):
                gi += 1
                attempt -= 1
                logger.warning("Groq refused word timings (%s); falling back to segment timings.", msg)
                continue
            raise TranscriptionError(f"Groq could not process this audio: {msg}")
        except groq.RateLimitError as e:
            last_err = e
            wait = 20.0 * attempt
            try:
                wait = max(wait, float(e.response.headers.get("retry-after", 0)))
            except Exception:
                pass
            logger.warning("Groq rate limit; sleeping %.0fs (attempt %d)", wait, attempt)
            time.sleep(min(wait, 120))
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
            last_err = e
            time.sleep(2.0 * attempt)
        except groq.APIStatusError as e:
            raise TranscriptionError(f"Groq API error ({e.status_code}): {getattr(e, 'message', e)}")
    raise TranscriptionError(
        "Transcription service is unavailable or rate-limited right now "
        f"(last error: {type(last_err).__name__}). Please retry in a minute."
    )


def _extra(resp: Any, name: str):
    v = _get(resp, name)
    if v is None:
        v = (_get(resp, "model_extra") or {}).get(name)
    return v


def _parse_words(resp: Any, offset: float) -> list[dict]:
    out = []
    for w in (_extra(resp, "words") or []):
        txt = (_get(w, "word", "") or "").strip()
        if txt:
            out.append({"start": offset + float(_get(w, "start", 0.0) or 0.0),
                        "end": offset + float(_get(w, "end", 0.0) or 0.0), "word": txt})
    return out


def _parse_segments(resp: Any, offset: float, chunk_len: float) -> list[dict]:
    raw = _extra(resp, "segments")
    out = []
    if raw:
        for s in raw:
            out.append({
                "start": offset + float(_get(s, "start", 0.0) or 0.0),
                "end": offset + float(_get(s, "end", 0.0) or 0.0),
                "text": (_get(s, "text", "") or "").strip(),
                "avg_logprob": _get(s, "avg_logprob"),
                "no_speech_prob": _get(s, "no_speech_prob"),
            })
    else:  # extremely defensive fallback: no segment info returned
        txt = (_get(resp, "text", "") or "").strip()
        if txt:
            out.append({"start": offset, "end": offset + chunk_len, "text": txt,
                        "avg_logprob": None, "no_speech_prob": None})
    return out


def _is_hallucination(seg: dict) -> bool:
    nsp, lp = seg.get("no_speech_prob"), seg.get("avg_logprob")
    return nsp is not None and lp is not None and nsp > 0.8 and lp < -1.0


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def transcribe(
    audio_path: str | Path,
    vocab_hint: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    progress_cb: ProgressCB = None,
) -> TranscriptionResult:
    """
    Transcribe a meeting recording.

    vocab_hint: optional comma-separated domain terms / names ("Kubernetes, Rahul, Q3 OKRs").
                Passed to Whisper as its `prompt`, which biases spelling of those words.
    """
    model = model or DEFAULT_MODEL
    report = progress_cb or (lambda f, m: None)

    path = validate_file(audio_path)
    _require_ffmpeg()
    client = _make_client(api_key)

    report(0.02, "Checking audio file...")
    duration = probe_audio(path)

    with tempfile.TemporaryDirectory(prefix="stt_") as tmp:
        tmpdir = Path(tmp)
        source = path
        if duration is None:  # e.g. some webm recordings: convert once to learn the length
            full = tmpdir / "full.flac"
            _to_flac(path, full)
            duration = probe_audio(full) or 0.0
            source = full
        if duration < MIN_DURATION_S:
            raise TranscriptionError("The audio is too short (under 1 second) to contain a meeting.")
        if duration > MAX_DURATION_S:
            raise TranscriptionError(
                f"The recording is {duration/3600:.1f} h long; the maximum supported length is 2 hours."
            )

        plans = plan_chunks(duration)
        prompt = None
        if vocab_hint and vocab_hint.strip():
            prompt = "Meeting vocabulary: " + vocab_hint.strip()

        kept: list[dict] = []
        kept_words: list[dict] = []
        got_words_all = True
        warnings: list[str] = []
        # work queue so that a chunk which is still too large after compression can be split in two
        queue = list(plans)
        done = splits = 0
        while queue:
            cut_s, cut_e, own_s, own_e = queue.pop(0)
            report(0.05 + 0.9 * done / (done + len(queue) + 1),
                   f"Transcribing part {done + 1}/{done + len(queue) + 1} "
                   f"({fmt_ts(own_s)}-{fmt_ts(own_e)})...")
            chunk_path = encode_chunk_for_upload(source, tmpdir, f"chunk_{done:03d}_{int(cut_s)}",
                                                 cut_s, cut_e - cut_s)
            if chunk_path is None:
                if own_e - own_s < 2 * MIN_SPLIT_SECONDS:
                    raise TranscriptionError("An audio chunk could not be compressed below the upload limit.")
                mid = (own_s + own_e) / 2
                queue[:0] = [(max(0.0, own_s - OVERLAP_SECONDS), min(duration, mid + OVERLAP_SECONDS), own_s, mid),
                             (max(0.0, mid - OVERLAP_SECONDS), min(duration, own_e + OVERLAP_SECONDS), mid, own_e)]
                splits += 1
                continue
            resp, got_words = _call_groq(client, chunk_path, model, prompt)
            got_words_all = got_words_all and got_words
            first_chunk, last_chunk = own_s <= 0.0, own_e >= duration

            def owned(a: float, b: float) -> bool:
                mid_ = (a + b) / 2
                return (first_chunk or mid_ >= own_s) and (last_chunk or mid_ < own_e)
            for s_ in _parse_segments(resp, cut_s, cut_e - cut_s):
                if owned(s_["start"], s_["end"]):
                    kept.append(s_)
            for w_ in _parse_words(resp, cut_s):
                if owned(w_["start"], w_["end"]):
                    kept_words.append(w_)
            done += 1
            try:
                chunk_path.unlink()                 # keep temp disk usage low on long recordings
            except OSError:
                pass

    if splits:
        warnings.append(f"Audio was sent in {done} smaller parts (re-encoded/split {splits}x) to fit the "
                        "speech-to-text upload limit; transcription quality is unaffected.")

    # clean-up: drop hallucinations, empties, and runaway repeats
    cleaned: list[Segment] = []
    repeat = 0
    prev_norm = None
    dropped = 0
    kept_words.sort(key=lambda w: w["start"])
    for s in sorted(kept, key=lambda x: x["start"]):
        text = s["text"]
        if not text or _is_hallucination(s):
            dropped += 1
            continue
        n = _norm(text)
        repeat = repeat + 1 if n == prev_norm else 0
        prev_norm = n
        if repeat >= 2:                      # same line 3+ times in a row = Whisper loop
            dropped += 1
            continue
        lp = s.get("avg_logprob")
        seg = Segment(round(s["start"], 2), round(s["end"], 2), text,
                      avg_logprob=float(lp) if lp is not None else None)
        if kept_words:
            seg.words = [Word(round(w["start"], 2), round(w["end"], 2), w["word"]) for w in kept_words
                         if seg.start - 0.05 <= (w["start"] + w["end"]) / 2 <= seg.end + 0.05]
        cleaned.append(seg)
    if dropped:
        warnings.append(f"Removed {dropped} low-confidence / repeated segment(s) likely caused by silence or noise.")

    if not cleaned:
        raise TranscriptionError(
            "No speech was detected in this recording. Check that the file contains audible English speech."
        )
    has_words = bool(kept_words) and got_words_all and any(s.words for s in cleaned)
    report(1.0, "Transcription complete.")
    return TranscriptionResult(
        segments=cleaned, duration_s=float(duration), model=model, chunks=done,
        has_word_timings=has_words, warnings=warnings,
    )


if __name__ == "__main__":      # quick manual test:  python transcriber.py meeting.mp3
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>\n(For the full app run:  streamlit run app.py)")
        raise SystemExit(1)
    try:
        res = transcribe(sys.argv[1], progress_cb=lambda f, m: print(f"{f:5.0%}  {m}"))
    except TranscriptionError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
    print(res.text())
