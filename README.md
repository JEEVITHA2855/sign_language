# OmniSign AI: Multimodal Sign Language & Speech-to-Text System

A real-time, multimodal communication system combining **Indian Sign Language (ISL) → Text**, **American Sign Language (ASL) → Text**, and **Speech → Text (Whisper ASR)** into a unified **Common Communication Layer** with text-to-speech (TTS) synthesis and web/desktop user interfaces.

---

## 1. System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │               INPUT STREAMS                  │
                    └──────────────────────────────────────────────┘
                             │                   │
                     Webcam Frame        Microphone Audio
                             │                   │
             ┌───────────────┴──────────────┐    │
             ▼                              ▼    ▼
      [ ISL PIPELINE ]              [ ASL PIPELINE ]       [ SPEECH PIPELINE ]
  ┌───────────────────────┐     ┌───────────────────────┐  ┌───────────────────────┐
  │ Alphabet:             │     │ Alphabet:             │  │ OpenAI Whisper ASR    │
  │ HF Image Classifier   │     │ MediaPipe 63-Landmark │  │ (openai/whisper-base) │
  │ (Hemg/ISL-Classifier) │     │ ONNX Dense Classifier │  │ 16kHz Float32 PCM     │
  │                       │     │                       │  │                       │
  │ Word:                 │     │ Word:                 │  │                       │
  │ Video Buffer (30 fps) │     │ WLASL I3D (32 frames) │  │                       │
  │ Custom Sequence Model │     │ 2,000 Sign Glosses    │  │                       │
  └───────────────────────┘     └───────────────────────┘  └───────────────────────┘
             │                              │                          │
             ▼                              ▼                          ▼
   Confidence Filter              Confidence Filter            Confidence Filter
  (Threshold: e.g. 55%)          (Threshold: e.g. 50%)        (Speech Conf Metric)
             │                              │                          │
             ▼                              ▼                          │
     Temporal Smoother              Temporal Smoother                  │
   (Majority/Stabilizer)          (Window Stabilization)               │
             │                              │                          │
             └──────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │       COMMON TEXT LAYER              │
                 │  - Source Attribution [ISL/ASL/VOICE]│
                 │  - Chronological Deduplication       │
                 │  - Combined Transcript Stream        │
                 └──────────────────┬───────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
           [ User Interface ]              [ Audio Synthesis ]
       - Modern Web UI (Glassmorphic)    - pyttsx3 / Windows SAPI
       - OpenCV HUD Dashboard            - Immediate Speech Out
```

---

## 2. Model Discovery & Validation Matrix

Pretrained models were researched, benchmarked, and selected based on licensing, local execution feasibility, and verified scopes:

| Pipeline | Model Name / Link | License | Dataset | Input Format | Output Format | Classes | Scope | Hardware / Real-Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ISL Alphabet** | [`Hemg/Indian-sign-language-classification`](https://huggingface.co/Hemg/Indian-sign-language-classification) | Apache-2.0 | ISL Alphabet Dataset | Single RGB Frame (224×224) | Class Probabilities | ~35 classes | Static ISL Alphabets (A-Z, digits) | CPU/GPU, ~15-30 ms latency |
| **ASL Alphabet** | Local ONNX Landmark Classifier | MIT | Custom Landmark Annotations | 63-dim Normalized Hand Coordinates | Softmax Logits | 28 classes (A-Z, SPACE, DEL) | Static Fingerspelling | Ultra-fast CPU (<1 ms latency, >10,000 FPS) |
| **ASL Word** | WLASL I3D PyTorch Model (`models/asl/`) | MIT | WLASL (World-Level ASL) | 32×224×224 RGB video clip | Top-1 Gloss Probability | 2,000 words | Isolated Dynamic Words | Real-time capable (GPU recommended, CPU ~150-250ms per word window) |
| **Speech-to-Text** | [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) | MIT | Multilingual Audio (680,000h) | 16 kHz Mono Audio PCM | Transcribed Text String | Continuous English/Multilingual | Sentence-level Continuous Speech | CPU/GPU (RTF < 0.2x, faster than real-time) |
| **Text-to-Speech** | `pyttsx3` / SAPI5 / NSSpeech | MIT | OS Voice Synthesizers | Plain Text String | Speaker Audio Out | N/A | Speech Synthesis | Instantaneous (<10 ms) |

> [!IMPORTANT]
> **No Imaginary Labels**: ISL static alphabet models do **not** claim continuous sentence recognition. Dynamic gestures are separated into word video buffers, and speech recognition handles continuous natural phrasing.

---

## 3. Core Modules & Pipelines

### 3.1 ISL Recognition Module (`src/recognition/isl_recognizer.py`)
- **Alphabet Mode**: Wraps `ISLAlphabetAdapter` using the Hugging Face vision pipeline (`Hemg/Indian-sign-language-classification`).
- **Word Mode**: Interfaces with `ISLModelAdapter` for multi-frame clips.
- **Confidence Filtering**: Emits `"UNCERTAIN"` when below the configurable threshold (`thresholds.isl`, default `0.55`).
- **Switching**: Dynamic switching between `ALPHABET` and `WORD` modes via `.set_mode()`.

### 3.2 ASL Recognition Module (`src/recognition/asl_recognizer.py`)
- **Fingerspelling Mode**: High-speed MediaPipe Hand Landmarker (`models/hand_landmarker.task`) → 63-dimensional normalized Cartesian vector → ONNX Runtime.
- **Word Mode**: 32-frame rolling buffer (`VideoBuffer`) → `SignSegmenter` (movement tracking) → WLASL I3D classifier → 2,000 word glosses.
- **Isolation**: ASL and ISL class dictionaries and models are completely separated.

### 3.3 Speech-to-Text Module (`src/speech/speech_recognizer.py` & `audio_capture.py`)
- Integrates `openai/whisper-base` via Hugging Face `transformers`.
- Cross-platform microphone capture using `sounddevice` with automated silence detection and ring-buffering.
- Outputs `RecognitionResult` objects identical to sign recognizers for seamless integration into the common layer.

### 3.4 Common Communication Layer (`src/communication/common_text_layer.py`)
- Converges outputs from ISL, ASL, and Speech into a unified stream:
  - Source attribution badges: `[ISL]`, `[ASL]`, `[SPEECH]`.
  - Temporal deduplication window: prevents repeating the same token within a configurable window (default 2.0s).
  - Subscriber callbacks (`add_listener`) for real-time WebSocket / SSE / UI feeds.
  - Complete conversational transcript generation.

### 3.5 Temporal Smoothing & Confidence Handling
- **Confidence Filter**: Any prediction with `confidence < threshold` is rejected and flagged as `"Sign not recognized. Please try again."`
- **Temporal Stabilizer**: Uses windowed majority voting (`window = 8`, `min_ratio = 0.65`) and debouncing to ensure hands are held in pose before accepting characters.
- **Word Stabilizer**: Consecutive agreement window for dynamic video signs.

---

## 4. User Interfaces

### 4.1 Modern Web Application (FastAPI & Glassmorphic UI)
- **Frontend**: Clean Vanilla CSS with Google Fonts (`Outfit`, `JetBrains Mono`), glowing ambient orbs, dark slate glassmorphism.
- **Interactive Controls**:
  - Language: `[ ISL ▼ ]` / `[ ASL ▼ ]`
  - Recognition Mode: `[ Alphabet (Static) ▼ ]` / `[ Sign / Word (Dynamic) ▼ ]`
  - Camera: `[ Start Camera ]` `[ Stop Camera ]`
  - Speech: `[ Start Listening ]` `[ Stop Listening ]`
  - Live HUD: Real-time FPS, Inference Latency (ms), and Confidence meter bar.
  - Common Text Stream: Live chat-style feed showing inputs from all sources.
  - TTS: Audio read-aloud button and settings modal with threshold sliders.
- **Server**: Zero-dependency fallback built into `src/api/server.py` ensuring it runs in any standard Python environment.

### 4.2 Desktop OpenCV Dashboard
- Real-time video window with annotated hand skeletons, keyboard shortcuts (`[G]` AI reconstruct, `[S]` TTS speak, `[M]` TTS mode, `[C]` clear, `[1/2]` mode switch), and alphabetical gesture guide.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.8+ (Python 3.8 to 3.11 recommended)
- Webcam and Microphone

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repo-url>
cd mini_project

# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate
# Activate on Linux/macOS
# source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Local Assets
Ensure the model files are present:
- `models/hand_landmarker.task`
- `models/gesture_classifier.onnx`
- `models/asl/pytorch_model.bin` (or `models/asl_download_test/`)

---

## 6. How to Run

### Option A: Launch Web Application (Recommended)
```bash
python main.py
```
Open your browser to: **`http://127.0.0.1:8000`**

### Option B: Launch Desktop OpenCV HUD Application
```bash
python main.py --opencv
```

### Option C: Run Headless / Camera Mock Test
```bash
python main.py --opencv --mock --frames 30 --no-tts
```

---

## 7. Evaluation & Metrics

Reproducible evaluation scripts are provided in `evaluation/`:

### 7.1 Run Benchmarks via CLI
```bash
# Run all benchmarks
python main.py --eval all

# Run individual benchmarks
python main.py --eval isl
python main.py --eval asl
python main.py --eval speech
```

### 7.2 Benchmark Output Examples
- **ASL Landmark Classifier**: Mean Latency: `0.086 ms`, Throughput: `11,649 FPS`.
- **WLASL Word Classifier**: Clip Duration: `32 frames`, Real-Time factor compliant.
- **Whisper Speech ASR**: Real-Time Factor (RTF): `< 0.2x` on CPU/GPU.
- **Evaluation Reports**: Saved automatically as JSON in `evaluation/*_eval_report.json`.

---

## 8. Training & Fine-Tuning Scripts

To fine-tune models on your own custom sign dataset:

### 8.1 Fine-Tune ISL Vision Classifier
```bash
python training/finetune_isl.py --data_dir path/to/dataset --epochs 10 --batch_size 16
```
- Features: PyTorch MobileNetV3 / Vision backbone, data augmentation (rotation, jitter), 80/20 train-val split, cosine annealing scheduler, checkpoint saving.

### 8.2 Train ASL Landmark Classifier
```bash
python training/finetune_asl.py --data path/to/landmarks.npz --output models/asl_custom.joblib
```
- Trains an MLP on 63-dimensional MediaPipe landmark arrays with validation accuracy and classification reports.

---

## 9. Project Directory Structure

```
mini_project/
├── config/
│   └── config.yaml               # Central configuration (thresholds, models, TTS)
├── evaluation/
│   ├── evaluate_asl.py           # ASL benchmarks & evaluation
│   ├── evaluate_isl.py           # ISL benchmarks & evaluation
│   └── evaluate_speech.py        # Speech WER / latency benchmarks
├── frontend/
│   ├── index.html                # Modern HTML5 Web UI
│   ├── style.css                 # Glassmorphic dark design system
│   └── app.js                    # WebRTC & client-side inference loop
├── models/
│   ├── gesture_classifier.onnx   # Static alphabet ONNX classifier
│   ├── hand_landmarker.task      # MediaPipe Hand landmark model
│   ├── asl/                      # WLASL I3D model checkpoint & labels
│   └── isl/                      # ISL fine-tuned weights & labels
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py             # FastAPI / HTTP REST & WebSocket API
│   ├── communication/
│   │   ├── __init__.py
│   │   └── common_text_layer.py  # Unified multi-modal communication layer
│   ├── recognition/
│   │   ├── asl_model_adapter.py  # WLASL I3D model adapter
│   │   ├── asl_recognizer.py     # ASL recognition pipeline
│   │   ├── base_recognizer.py    # Common recognition interface
│   │   ├── isl_alphabet_adapter.py # Hugging Face ISL image classifier adapter
│   │   ├── isl_model_adapter.py  # ISL video adapter
│   │   ├── isl_recognizer.py     # ISL dual-mode recognition pipeline
│   │   └── recognition_result.py # Standardized prediction dataclass
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── audio_capture.py      # sounddevice microphone capture
│   │   └── speech_recognizer.py  # OpenAI Whisper ASR recognizer
│   ├── app.py                    # OpenCV desktop HUD dashboard
│   ├── classifier.py             # Landmark classifier wrapper
│   ├── landmark_extractor.py     # MediaPipe hand tracker
│   ├── stabilizer.py             # Temporal smoothing & debouncer
│   ├── text_composer.py          # Word & sentence buffer
│   └── tts.py                    # Text-to-speech audio engine
├── tests/
│   ├── test_multimodal_system.py # End-to-end multimodal test suite
│   └── ...                       # Component unit tests
├── training/
│   ├── finetune_asl.py           # ASL training script
│   └── finetune_isl.py           # ISL fine-tuning script
├── main.py                       # CLI & application entrypoint
├── requirements.txt              # Complete dependency specifications
└── README.md                     # Comprehensive documentation
```

---

## 10. Summary of Key Constraints & Best Practices

1. **Model Discovery First**: Verified Hugging Face `Hemg/Indian-sign-language-classification`, `openai/whisper-base`, and local `WLASL I3D`.
2. **Distinct Class Mappings**: ISL and ASL class mappings are strictly isolated and never conflated.
3. **Graceful Fallbacks**: All network-dependent models and hardware devices fallback gracefully without application crashes.
4. **Configurable Confidence**: Never displays raw uncertain guesses; alerts users to retry when confidence is below the threshold.
5. **Aesthetic Excellence**: Built with a responsive, modern glassmorphic web dashboard providing live feedback.
