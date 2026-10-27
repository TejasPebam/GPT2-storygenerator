import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"

st.set_page_config(page_title="GPT-2 Story Studio", page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0b1020 0%, #141a2e 55%, #101827 100%); color: #f3f5ff; }
.block-container { max-width: 1180px; padding-top: 2rem; }
.hero { padding: 1.6rem 1.8rem; border: 1px solid #303a58; border-radius: 22px;
        background: linear-gradient(120deg, rgba(79,70,229,.23), rgba(14,165,233,.10)); margin-bottom: 1.2rem; }
.hero h1 { margin: 0; font-size: 2.35rem; color: #fff; }
.hero p { color: #b9c5e5; margin-bottom: 0; }
.panel { padding: 1.2rem; border: 1px solid #303a58; border-radius: 18px; background: rgba(22,30,51,.78); }
.story { white-space: pre-wrap; line-height: 1.8; color: #e8edff; font-size: 1.02rem; }
.small-note { color: #aab7d6; font-size: .9rem; }
div[data-testid="stMetric"] { background: rgba(30,41,68,.8); border: 1px solid #303a58; padding: 12px; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📚 GPT-2 Story Studio</h1>
  <p>Turn one opening sentence into a story. Experiment with writing styles and temperature.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def style_instruction(style):
    styles = {
        "Adventure": "Write in an exciting adventure style.",
        "Mystery": "Write a suspenseful mystery with clues and an intriguing atmosphere.",
        "Fantasy": "Write an imaginative fantasy story with magical details.",
        "Science fiction": "Write a science-fiction story with futuristic technology.",
        "Horror": "Write a creepy horror story with suspense, but no graphic violence.",
        "Romance": "Write a warm romantic story focused on emotions and relationships.",
        "Comedy": "Write a light-hearted, funny story with playful situations.",
        "General": "Continue this as an engaging story."
    }
    return styles.get(style, styles["General"])

def generate_story(tokenizer, model, opening, style, temperature, max_new_tokens, top_k, seed):
    torch.manual_seed(seed)
    prompt = f"{style_instruction(style)}\nStory opening: {opening.strip()}\nContinue the story:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(float(temperature), 0.05),
            top_k=int(top_k),
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3
        )
    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    story = opening.strip()
    if continuation:
        story += "\n\n" + continuation
    total_tokens = int(output.shape[1])
    story_tokens = len(tokenizer.encode(story))
    return story, story_tokens, total_tokens

with st.sidebar:
    st.markdown("## ⚙️ Story settings")
    style = st.selectbox("Writing style", ["Adventure", "Mystery", "Fantasy", "Science fiction", "Horror", "Romance", "Comedy", "General"])
    temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.8, step=0.1,
                             help="Lower values are more predictable; higher values are more varied and may be less coherent.")
    max_new_tokens = st.slider("Story length (new tokens)", 40, 220, 100, 10)
    top_k = st.slider("Top-k sampling", 0, 100, 40, 10, help="0 disables top-k filtering.")
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    st.caption("Model: GPT-2 (124M parameters)")
    st.caption("Model weights run locally after download. First launch needs an internet connection to download them.")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown("### ✍️ Start your story")
    opening = st.text_area("Opening sentence", value="At midnight, the old lighthouse began to shine again.", height=130,
                           placeholder="Enter the first sentence of your story...")
    st.markdown('<p class="small-note">Tip: A specific opening sentence usually gives the model a clearer direction.</p>', unsafe_allow_html=True)
    generate = st.button("✨ Generate story", type="primary", use_container_width=True)

with right:
    st.markdown("### 🧪 What temperature changes")
    st.markdown("""
    <div class="panel">
    <b>Low temperature</b><br><span class="small-note">Usually safer and more predictable wording.</span><br><br>
    <b>High temperature</b><br><span class="small-note">Usually more varied wording, with a higher chance of strange or inconsistent text.</span><br><br>
    <b>Autoregressive generation</b><br><span class="small-note">GPT-2 predicts the next token repeatedly, using the tokens already generated as context.</span>
    </div>
    """, unsafe_allow_html=True)

if generate:
    if not opening.strip():
        st.warning("Please enter an opening sentence first.")
    else:
        try:
            with st.spinner("Loading GPT-2 and generating your story… first run may download the model weights."):
                tokenizer, model = load_model()
                story, story_tokens, total_tokens = generate_story(
                    tokenizer, model, opening, style, temperature, max_new_tokens, top_k, int(seed)
                )
            st.session_state["story_result"] = {
                "story": story, "story_tokens": story_tokens, "total_tokens": total_tokens,
                "style": style, "temperature": temperature
            }
        except Exception as e:
            st.error(f"Could not load or run GPT-2: {e}")
            st.info("Check your internet connection on first launch and install the packages from requirements.txt.")

if "story_result" in st.session_state:
    result = st.session_state["story_result"]
    st.markdown("---")
    st.markdown("## 📖 Generated story")
    st.caption(f"Style: {result['style']}  •  Temperature: {result['temperature']:.1f}")
    st.markdown('<div class="panel"><div class="story">' + result["story"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + '</div></div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Story tokens", result["story_tokens"])
    m2.metric("Output sequence tokens", result["total_tokens"])
    m3.metric("Temperature", f"{result['temperature']:.1f}")

    st.download_button("⬇️ Download story as TXT", data=result["story"], file_name="generated_story.txt", mime="text/plain")

st.markdown("---")
st.markdown("## 🔬 Compare two temperatures")
st.write("Generate two continuations from the same opening sentence to observe how sampling temperature changes the output.")
c1, c2 = st.columns(2)
with c1:
    temp_a = st.slider("Temperature A", 0.2, 1.5, 0.4, 0.1, key="temp_a")
with c2:
    temp_b = st.slider("Temperature B", 0.2, 1.5, 1.2, 0.1, key="temp_b")
compare = st.button("⚖️ Generate comparison", use_container_width=True)

if compare:
    if not opening.strip():
        st.warning("Please enter an opening sentence before comparing.")
    else:
        try:
            with st.spinner("Generating both versions with the local GPT-2 model…"):
                tokenizer, model = load_model()
                story_a, tokens_a, seq_a = generate_story(tokenizer, model, opening, style, temp_a, max_new_tokens, top_k, int(seed))
                story_b, tokens_b, seq_b = generate_story(tokenizer, model, opening, style, temp_b, max_new_tokens, top_k, int(seed) + 1)
            st.session_state["comparison"] = (story_a, tokens_a, seq_a, story_b, tokens_b, seq_b, temp_a, temp_b)
        except Exception as e:
            st.error(f"Comparison failed: {e}")

if "comparison" in st.session_state:
    a, ta, sa, b, tb, sb, t_a, t_b = st.session_state["comparison"]
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown(f"### Version A — temperature {t_a:.1f}")
        st.markdown('<div class="panel"><div class="story">' + a.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + '</div></div>', unsafe_allow_html=True)
        st.metric("Story tokens", ta)
    with cb:
        st.markdown(f"### Version B — temperature {t_b:.1f}")
        st.markdown('<div class="panel"><div class="story">' + b.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + '</div></div>', unsafe_allow_html=True)
        st.metric("Story tokens", tb)
    st.caption(f"Output sequence tokens (including prompt) — A: {sa} · B: {sb}")
    st.info("A higher temperature changes the probability distribution used to sample the next token. It does not guarantee a better or longer story.")

with st.expander("🧠 Assignment notes / concepts"):
    st.markdown("""
    - **Tokenizer:** converts text into GPT-2 token IDs; token count is not the same as word count.
    - **Autoregressive generation:** GPT-2 predicts one token at a time, then uses the new token as context for the next prediction.
    - **Temperature:** scales next-token logits before sampling. Lower values make the distribution sharper; higher values flatten it.
    - **Top-k sampling:** restricts sampling to the k most likely next tokens.
    - **Local model constraint:** `from_pretrained("gpt2")` downloads the model weights to the local Hugging Face cache on first run; generation then runs on the computer, not through a hosted generation API.
    """)
