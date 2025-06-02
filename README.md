# 🎙️ Speaker Diarization with PyAnnote

This is a simple speaker diarization pipeline built using [`pyannote-audio`](https://github.com/pyannote/pyannote-audio). It processes a `.mp4` video file to figure out **who spoke when**, leveraging Hugging Face-hosted pretrained models.

---

##  Folder Layout

```
speaker_diarization_project/
│
├── main.py                    # Entry point — runs the full pipeline
├── requirements.txt           # List of required Python packages
├── README.md                  # Project documentation (this file)
│
├── utils/                     # Helper modules
│   ├── __init__.py
│   ├── audio_extractor.py     # Converts MP4 to mono WAV (16kHz)
│   └── diarizer.py            # Runs the diarization and saves plots
│
├── data/                      # Place your input .mp4 file here
│   └── input_video.mp4
│
└── outputs/                   # Outputs (audio + speaker timeline plot)
    ├── audio.wav
    └── diarization_plot.png
```

---

##  Getting Started

### 1. Set up your local environment

After cloning or unzipping this folder, navigate to it:

```bash
cd speaker_diarization_project
```

Optionally, set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # For Windows: venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Hugging Face Access

The models used in this project are gated. You’ll need a Hugging Face token and must manually agree to the model access terms.

### Here's what you need to do:

1. Get a Hugging Face token:  
   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
   Create a new one with **read** access.

2. Accept model access:  
   - [`pyannote/speaker-diarization`](https://huggingface.co/pyannote/speaker-diarization)  
   - [`pyannote/segmentation`](https://huggingface.co/pyannote/segmentation)

3. Once done, paste your token inside `main.py`:

```python
HUGGINGFACE_TOKEN = "your_token_here"
```

---

##  Running the Pipeline

1. Drop your video file into the `data/` folder and rename it to `input_video.mp4`.
2. Then just run:

```bash
python main.py
```

---

##  What You’ll Get

- Extracted audio (mono, 16kHz) saved as → `outputs/audio.wav`
- Speaker diarization timeline plot → `outputs/diarization_plot.png`
- Console output showing timestamps and speaker labels


