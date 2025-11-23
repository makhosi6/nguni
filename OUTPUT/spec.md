# 📘 **Whisper v3 Dataset Standardization Specification**

### Version 2.0 (Aligned with HF spec: [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3))

This specification defines requirements for converting multilingual audio + transcripts into a standardized dataset suitable for fine-tuning **Whisper large-v3** or **Whisper small** on HuggingFace.

---

# 1. **Final Required Dataset Format**

Whisper models expect the following per-example fields:

### **HF Whisper JSONL line format**

Each item must be:

```json
{
  "audio": {
    "path": "<relative-or-absolute-path>",
    "array": null,
    "sampling_rate": 16000
  },
  "text": "<clean transcript>",
  "language": "<iso-639-1-or-639-3-code>"
}
```

You may omit `"array"` and `"sampling_rate"` if you allow HF Datasets to load/resample automatically.

**Minimum valid example:**

```json
{
  "audio": "xh-ZA/audio/xh-ZA_train_spk01_00023.wav",
  "text": "Ndicela undincede ngale ngxelo.",
  "language": "xh"
}
```

---

# 2. **Final Required File Types**

Your current dataset has transcripts in:

* `.xml`
* `.csv`
* `.tsv`

All must be **converted → JSONL** before training.

---

# 3. **Directory Structure Requirements**

Keep existing layout:

```
<LANG>-ZA/
    audio/
    transcriptions/
    metadata/
    ...
```

After processing, each language directory must contain:

```
transcriptions/train.jsonl
transcriptions/val.jsonl
```

**AND optionally:**
A combined multilingual file:

```
dataset.jsonl
```

---

# 4. **Audio Standardization**

### ✔ Required by Whisper (HF standard)

| Parameter      | Requirement          |
| -------------- | -------------------- |
| Sample rate    | **16,000 Hz**        |
| Channels       | **mono (1 channel)** |
| Encoding       | PCM16 WAV preferred  |
| File duration  | Recommended ≤ 30 s   |
| File extension | `.wav` or `.flac`    |

### Allowed formats (must be converted)

* `.mp3`
* `.m4a`
* `.ogg`

**Required conversion command**:

```
ffmpeg -i input -ar 16000 -ac 1 output.wav
```

---

# 5. **Transcript Standardization**

All transcript sources (**XML, CSV, TSV**) must be normalized into the **Whisper JSONL schema**.

## 5.1 Allowed Input Transcript Formats

### 1. **XML**

Example input:

```xml
<utterance speaker="spk01" start="0.0" end="2.1">
    Hello, this is a test.
</utterance>
```

You must extract:

* text (node content)
* utterance start/end timestamps (ignored for training)
* speaker ID (optional)

### 2. **CSV**

Required columns (minimum one must exist):

* `text` OR `transcript` OR `utterance`
* `audio_file` OR `filename`

Optional:

* `speaker_id`
* `start`, `end`

### 3. **TSV**

Same requirements as CSV but tab-delimited.

---

## 5.2 Transcript Cleaning Rules (strict required)

Whisper requires **raw, natural, clean text**.

### You must remove:

* timestamps (00:00 → remove)
* `<tags>`, XML markup
* `[noise]`, `[laugh]` (unless used intentionally)
* uppercase labels: “SPEAKER 1:”
* disfluencies unless meaningful (“uh”, “um”)

### You must preserve:

* punctuation native to the language
* accents (à, ê, í, ḍ, etc.)
* natural sentence casing **OR** full lowercase
* numbers as spoken (preferred)

### You must standardize casing:

Choose **one**:

#### Option A — *Lowercase everything* (recommended)

```
“Ndicela undincede ngale ngxelo.” → “ndicela undincede ngale ngxelo.”
```

#### Option B — Keep original casing

Apply consistently per language.

---

# 6. **Language Codes (required by Whisper)**

Use ISO codes:

| Folder | Whisper code |
| ------ | ------------ |
| af-ZA  | af           |
| en-ZA  | en           |
| nr-ZA  | nr           |
| nso-ZA | nso          |
| ss-ZA  | ss           |
| st-ZA  | st           |
| tn-ZA  | tn           |
| ts-ZA  | ts           |
| ve-ZA  | ve           |
| xh-ZA  | xh           |
| zu-ZA  | zu           |

---

# 7. **Final JSONL Output Requirements**

Each JSONL line must include:

```json
{
  "audio": "<path>",
  "text": "<cleaned transcript>",
  "language": "<lang>"
}
```

### Example (Xhosa)

```json
{"audio": "xh-ZA/audio/seg_00012.wav", "text": "ndicela undincede.", "language": "xh"}
```

### Example (Afrikaans)

```json
{"audio": "af-ZA/audio/seg_00098.wav", "text": "ek sal môre terugkom.", "language": "af"}
```

---

# 8. **Train/Val Split Specification**

### Required:

* **train.jsonl**
* **val.jsonl**

### Recommended split:

* 90% train
* 10% val

Ensure files referenced in JSONL match split.

---

# 9. **Processing Workflow (Mandatory Pipeline)**

### Step 1 — Scan all audio files

List files in:
`<LANG>-ZA/audio/`

### Step 2 — Load transcripts

Parse:

* XML → extract text
* CSV → parse rows
* TSV → parse rows

Detect missing transcript → flag.

### Step 3 — Clean text

Apply rules in Section 5.

### Step 4 — Normalize audio

Convert to:

```
16 kHz / mono / WAV
```

### Step 5 — Generate JSONL lines

For each pair:

```json
{"audio": "path.wav", "text": "...", "language": "xh"}
```

### Step 6 — Export train & val JSONL

---

# 10. **Validation Requirements**

Every JSONL entry must pass:

### Audio validation

* File exists
* Duration > 0.3 s
* Successfully decodes

### Transcript validation

* `text` is non-empty
* Language matches directory
* No markup or timestamps remain

### Format validation

* Valid JSON
* `audio` string path exists

---

# 11. **Deliverables Produced by This Spec**

Each language directory will contain:

```
<LANG>-ZA/audio/*.wav
<LANG>-ZA/transcriptions/train.jsonl
<LANG>-ZA/transcriptions/val.jsonl
```

And optionally:

```
dataset.jsonl
```

---

# 12. **Optional Extras**

I can also generate:

* ✔ Complete Python conversion scripts (XML → JSONL, CSV/TSV → JSONL)
* ✔ Audio normalization script
* ✔ Full HuggingFace training script
* ✔ Dataset validator script
