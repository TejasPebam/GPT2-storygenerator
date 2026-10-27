import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const $ = (id) => document.getElementById(id);
const status = $("modelStatus");
const dot = $("statusDot");
const loadBtn = $("loadBtn");
const generateBtn = $("generateBtn");
const compareBtn = $("compareBtn");
const downloadBtn = $("downloadBtn");
let generator = null;
let latestStory = "";
let isBusy = false;

const stylePrompts = {
  Adventure: "Write an exciting adventure story.",
  Mystery: "Write a suspenseful mystery story with clues and an intriguing atmosphere.",
  Fantasy: "Write an imaginative fantasy story with magical details.",
  "Science fiction": "Write a science-fiction story involving futuristic technology.",
  Horror: "Write a creepy horror story with suspense, without graphic violence.",
  Romance: "Write a warm romantic story focused on emotions and relationships.",
  Comedy: "Write a light-hearted, funny story with playful situations.",
  General: "Continue this as an engaging story."
};

function setStatus(message, state = "") {
  status.textContent = message;
  dot.className = "dot" + (state ? ` ${state}` : "");
}

function setBusy(value, message) {
  isBusy = value;
  loadBtn.disabled = value || !!generator;
  generateBtn.disabled = value || !generator;
  compareBtn.disabled = value || !generator;
  if (message) setStatus(message, value ? "busy" : (generator ? "ready" : ""));
}

function readSettings(tempOverride = null, seedOffset = 0) {
  return {
    opening: $("opening").value.trim(),
    style: $("style").value,
    temperature: tempOverride ?? Number($("temperature").value),
    maxNewTokens: Number($("length").value),
    topK: Number($("topK").value),
    seed: Math.max(0, Number($("seed").value) || 0) + seedOffset
  };
}

function buildPrompt(opening, style) {
  return `${stylePrompts[style]}\nStory opening: ${opening}\nContinue the story:`;
}

function tokenCount(text) {
  try {
    const encoded = generator.tokenizer(text);
    const ids = encoded?.input_ids;
    if (ids?.data) return ids.data.length;
    if (Array.isArray(ids)) return Array.isArray(ids[0]) ? ids[0].length : ids.length;
    if (ids?.length != null) return ids.length;
  } catch (e) {
    console.warn("Could not count tokens:", e);
  }
  return "—";
}

async function ensureModel() {
  if (generator) return generator;
  setBusy(true, "Downloading and loading GPT-2 model… first load can take a while.");
  loadBtn.textContent = "Loading model…";
  try {
    // All inference is performed in this browser tab; no hosted text-generation API is used.
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    generator = await pipeline("text-generation", "Xenova/gpt2", {
      dtype: "q8",
      progress_callback: (progress) => {
        if (progress?.status === "progress" && progress.total) {
          const percent = Math.round((progress.loaded / progress.total) * 100);
          setStatus(`Downloading model files… ${percent}%`, "busy");
        } else if (progress?.status === "downloading") {
          setStatus(`Downloading ${progress.file ?? "model files"}…`, "busy");
        } else if (progress?.status === "initiate") {
          setStatus(`Preparing ${progress.file ?? "model files"}…`, "busy");
        }
      }
    });
    setStatus("GPT-2 is loaded and ready in this browser.", "ready");
    generateBtn.disabled = false;
    compareBtn.disabled = false;
    loadBtn.textContent = "Model loaded ✓";
    loadBtn.disabled = true;
    return generator;
  } catch (error) {
    generator = null;
    setStatus("Model failed to load. Check your internet connection and reload.", "error");
    loadBtn.textContent = "Retry loading GPT-2";
    loadBtn.disabled = false;
    throw error;
  } finally {
    isBusy = false;
    if (generator) {
      generateBtn.disabled = false;
      compareBtn.disabled = false;
    }
  }
}

async function generate(settings) {
  const prompt = buildPrompt(settings.opening, settings.style);
  const options = {
    max_new_tokens: settings.maxNewTokens,
    do_sample: true,
    temperature: Math.max(0.05, settings.temperature),
    top_k: settings.topK || 0,
    repetition_penalty: 1.12,
    return_full_text: false,
    seed: settings.seed
  };
  const result = await generator(prompt, options);
  let continuation = result?.[0]?.generated_text ?? "";
  if (typeof continuation !== "string") continuation = String(continuation);
  if (continuation.startsWith(prompt)) continuation = continuation.slice(prompt.length);
  continuation = continuation.trim();
  const story = settings.opening + (continuation ? "\n\n" + continuation : "");
  return {
    story,
    storyTokens: tokenCount(story),
    outputTokens: tokenCount(prompt + continuation)
  };
}

function showStory(text, empty = false) {
  const output = $("storyOutput");
  output.textContent = text;
  output.classList.toggle("empty", empty);
}

function showError(error) {
  console.error(error);
  const message = error?.message || String(error);
  setStatus("Something went wrong. See the message below.", "error");
  const old = document.querySelector(".notice.error");
  if (old) old.remove();
  const notice = document.createElement("div");
  notice.className = "notice error";
  notice.textContent = `Generation error: ${message}. Try reloading the page. On the first run, allow the model files to finish downloading.`;
  $("storyOutput").before(notice);
}

async function onGenerate() {
  const settings = readSettings();
  if (!settings.opening) {
    $("opening").focus();
    showError(new Error("Please enter an opening sentence."));
    return;
  }
  if (isBusy) return;
  document.querySelectorAll(".notice.error").forEach((node) => node.remove());
  setBusy(true, "Generating story locally…");
  generateBtn.textContent = "Generating…";
  try {
    await ensureModel();
    setBusy(true, "Generating story locally…");
    const result = await generate(settings);
    latestStory = result.story;
    showStory(result.story);
    $("storyTokens").textContent = result.storyTokens;
    $("outputTokens").textContent = result.outputTokens;
    $("usedTemp").textContent = settings.temperature.toFixed(1);
    downloadBtn.disabled = false;
    setStatus("Story generated successfully.", "ready");
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
    generateBtn.textContent = "✦ Generate story";
  }
}

async function onCompare() {
  const settings = readSettings();
  if (!settings.opening) {
    $("opening").focus();
    showError(new Error("Please enter an opening sentence before comparing."));
    return;
  }
  if (isBusy) return;
  document.querySelectorAll(".notice.error").forEach((node) => node.remove());
  const tempA = Number($("tempA").value);
  const tempB = Number($("tempB").value);
  setBusy(true, "Generating two local GPT-2 versions…");
  compareBtn.textContent = "Generating comparison…";
  try {
    await ensureModel();
    setBusy(true, "Generating version A…");
    const a = await generate({ ...settings, temperature: tempA, seed: settings.seed });
    $("outputA").textContent = a.story;
    $("outputA").classList.remove("empty");
    $("tokensA").textContent = a.storyTokens;
    $("labelA").textContent = `· ${tempA.toFixed(1)}`;
    setStatus("Generating version B…", "busy");
    const b = await generate({ ...settings, temperature: tempB, seed: settings.seed + 1 });
    $("outputB").textContent = b.story;
    $("outputB").classList.remove("empty");
    $("tokensB").textContent = b.storyTokens;
    $("labelB").textContent = `· ${tempB.toFixed(1)}`;
    setStatus("Temperature comparison complete.", "ready");
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
    compareBtn.textContent = "⚖ Generate comparison";
  }
}

loadBtn.addEventListener("click", async () => {
  if (isBusy || generator) return;
  document.querySelectorAll(".notice.error").forEach((node) => node.remove());
  try {
    await ensureModel();
  } catch (error) {
    showError(error);
  } finally {
    isBusy = false;
    loadBtn.disabled = !!generator;
    generateBtn.disabled = !generator;
    compareBtn.disabled = !generator;
  }
});
generateBtn.addEventListener("click", onGenerate);
compareBtn.addEventListener("click", onCompare);
downloadBtn.addEventListener("click", () => {
  if (!latestStory) return;
  const blob = new Blob([latestStory], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "generated_story.txt";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
});
$("temperature").addEventListener("input", () => $("tempValue").textContent = Number($("temperature").value).toFixed(1));
$("tempA").addEventListener("input", () => {
  $("tempAValue").textContent = Number($("tempA").value).toFixed(1);
  $("labelA").textContent = `· ${Number($("tempA").value).toFixed(1)}`;
});
$("tempB").addEventListener("input", () => {
  $("tempBValue").textContent = Number($("tempB").value).toFixed(1);
  $("labelB").textContent = `· ${Number($("tempB").value).toFixed(1)}`;
});
