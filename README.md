# GPT-2 Story Studio — Netlify deployment package

This is a static, Netlify-compatible version of the GPT-2 Interactive Story Generator assignment.

## Important architecture difference
Netlify static hosting does not run a Streamlit/Python server. This package therefore uses the browser-based Hugging Face Transformers.js runtime with the downloadable `Xenova/gpt2` model. The model weights are downloaded from Hugging Face on first use and inference runs in the visitor's browser. It does not call a hosted text-generation API.

## Deploy on Netlify
1. Extract this ZIP.
2. In Netlify, choose **Add new site → Deploy manually** and drag the extracted folder's contents (or the folder, if the UI accepts it) into the deploy drop area.
3. Alternatively, push the contents to GitHub and import that repository into Netlify.
4. The publish directory is the project root (`.`); `netlify.toml` already sets this.
5. Open the generated `*.netlify.app` URL and click **Load GPT-2 model**.

## Requirements and first launch
- A modern browser and internet access are required to load the JS runtime and model files.
- The first load can take several minutes and downloads model weights into the browser cache.
- The model is relatively large; hosting does not bundle model weights inside this ZIP.
- Generation speed and memory use depend on the visitor's device and browser.
- This package cannot guarantee error-free execution on every device/network; use browser DevTools Console if a runtime error appears.

## Included features
- Opening sentence, writing-style selector, temperature control
- Maximum new tokens and top-k sampling controls
- Random seed input
- Generated story and token counts
- Two-temperature comparison
- Download generated story as TXT
- Responsive layout for desktop and mobile

## Assignment note
The original Streamlit Python app remains a separate local-run version. For a strict requirement that the Python `GPT2LMHeadModel` run on a server, Netlify static hosting alone is not the right platform; use a Python-capable host instead.
