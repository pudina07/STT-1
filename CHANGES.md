# What changed in this version

## 1. Confidence tiers for decisions and action items (new Stage 4b, `confidence.py`)
LLM #2 still decides *what* the decisions/tasks are, but it no longer has the last word on *how sure* we are.
It now also reports **evidence features** it can read off the text:

| item | features |
|---|---|
| decision | `agreement_signal` (explicit / implicit / none), `objection` (none / resolved / unresolved), `hedged` |
| action item | `acceptance` (explicit / implicit / none), `hedged` |

A deterministic scorer (no LLM) combines them with hard evidence from the pipeline into a 0–100 score:
- the evidence quote is really in the transcript (an invented quote is capped at 30, i.e. Low chance)
- lexical cues in the quote and the next few turns: formal settlement ("motion carried", "that's settled"), assent, push-back ("I'm not convinced", "hold on"), hedges ("maybe", "probably"), first-person commitments ("I'll …"). These **cross-check the LLM**: an "explicit" claim with no agreement wording nearby is downgraded to implicit.
- acknowledgement by a *different* speaker (when diarization is on)
- **ASR clarity** of the passage (Whisper `avg_logprob`, now kept per paragraph)
- whether the refiner changed words inside the quote

| tier | score | meaning |
|---|---|---|
| 🟢 Confirmed | ≥ 80 | explicitly settled / accepted, verified quote, clear audio |
| 🔵 High chance | 60–79 | agreed or accepted, but implicitly or with one weaker signal |
| 🟠 Ambiguous | 40–59 | mixed signals: push-back, hedging, unclear audio, or a well-supported proposal |
| 🔴 Low chance | < 40 | only suggested or weakly evidenced |

The scorer also enforces hard rules so a score can never contradict the problem statement: proposals are capped at Ambiguous, "Confirmed" needs explicit settlement (or explicit acceptance by a stated owner), an agreed decision with an open objection is forced to Ambiguous, unclear audio caps an item at High chance, and scores never go above 97%. Every point added or removed is stored as a readable reason ("Why this score?").

## 2. Timestamps on every decision and action item
- Paragraph timings and clarity are carried from the transcriber through the refiner (`RefinedTurn.start/end/asr_conf`).
- Each quote is located in the transcript and its start time is interpolated inside the paragraph. Decisions get a statement timestamp and a separate **settlement timestamp**.
- Streamlit: a **review player** in the sidebar (always visible). Every ▶ button (decisions, tasks, and each paragraph in "Listen by paragraph") jumps the player to 2 s before that point. It plays a small 48 kbps copy of the recording (about 9 MB for 25 min).
- New tab **"5 · Decisions & tasks review"** with tier filter, confidence bar, quotes with times, and listen buttons.
- Transcripts can be shown or downloaded with `[mm:ss]` prefixes. Minutes markdown and JSON include `timestamp`, `confidence`, `confidence_score`, `confidence_reasons` and `clarity`.

## 3. Large-file fix (25-min recordings)
- Root cause: Streamlit's default **200 MB upload cap**. A 25-min WAV is about 260 MB, so the upload was refused before the pipeline ran. `.streamlit/config.toml` raises it to 2 GB. **Run `streamlit run app.py` from this folder** so the config is picked up.
- Uploads are streamed to disk instead of copied in memory.
- Adaptive chunking: each 10-min chunk is size-checked. If the FLAC is over the STT limit it is re-encoded to 64 then 32 kbps MP3 (or Opus), and as a last resort split in half, recursively. Overlap ownership guarantees no duplicate or missing text.
- New "use a file already on this machine" path input, so very large files skip the browser upload entirely.
- Tested on a 25-min, 264 MB stereo WAV: 3 chunks of 7–14 MB with the default limit, and 10 parts when forced to a 1 MB limit, with identical segments in both runs.
