# GPT-2 Interactive Story Generator

A local GPT-2 story generation app built with Streamlit and Hugging Face Transformers.

## Features
- Opening sentence input
- Writing style selector
- Temperature slider
- Local GPT-2 model loading
- Generated story and token counts
- Compare two temperatures
- Top-k sampling and random seed controls
- Download generated story as TXT

## Run on Windows
1. Install Python 3.10 or 3.11.
2. Extract this ZIP.
3. Open Command Prompt in the extracted folder.
4. Run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## Important
The first run downloads the public GPT-2 model weights from Hugging Face into the local cache, so internet access is needed once. The story generation itself runs locally; this app does not call a hosted text-generation API. GPT-2 is a relatively small older model, so some outputs may be repetitive or inconsistent.
