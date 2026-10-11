# Sinhal Joshi

**I build and evaluate LLM agent systems** — retrieval pipelines, agent tooling, and the evals that tell you whether any of it actually works.

Most of what I publish falls into one of two buckets: systems that retrieve and reason over a real corpus, and harnesses that measure where a model fails. The second is the part I find more interesting, and it's where my strongest work is.

---

## Selected work

### [laya-zeroshot-agent-guard-eval](https://github.com/Zephyr4772) — can a small model act as an agent step-guard?

61 hand-written agent transcripts × 4 decisions = **244 judgments** across 3 checkpoints.

What it found:

| Failure mode | Result |
|---|---|
| Believes the agent's `"done"` over tool output showing failures | **0 / 6** |
| Never escalates destructive commands (`DELETE FROM users;` → *continue*) | **0 / 8** |
| Continue/stop score is inverted — flipping it would score better | **AUROC 0.32** |

Reported with bootstrapped confidence intervals and a majority-class baseline that beats all three checkpoints.

`python` · `transformers` · `evaluation`

---

### [Lex.ai](https://github.com/Zephyr4772) — RAG over 41,876 Indian Supreme Court judgments

Local ChromaDB index, `nomic-embed-text` embeddings via Ollama, streaming analysis from Gemini 2.5-flash with automatic fallback to a local model. Retrieval stays on-device — only retrieved passages reach the LLM. Every claim renders as a traceable case card.

The judgment corpus is published as an open dataset: **[Indian_Supreme_Court_Judgments](https://huggingface.co/datasets/sinhal/Indian_Supreme_Court_Judgments)**

`fastapi` · `chromadb` · `ollama` · `react`

---

### [ShipScript](https://github.com/Zephyr4772) — an MCP server that documents your repo for you

Analyzes a GitHub repo, detects the stack, screenshots the live deployment, generates charts, and drafts a README plus launch posts. Exposed as MCP tools, with a Gradio UI.

`mcp` · `litellm` · `playwright` · `gradio`

---

### [Hushiyar](https://ollama.com/aryanvala/hushiyar-alpha) — an 8B language model pretrained from scratch

~4TB of cleaned text, ~4 days of continuous distributed training across 8 rented NVIDIA H200s, trained to the point of diminishing gains. Shipped with a ~1B variant (decoder-only, RoPE, 32K context) for edge inference and a .NET chat application.

Built with one teammate as a diploma capstone. [8B](https://ollama.com/aryanvala/hushiyar-alpha) · [1B](https://ollama.com/aryanvala/hushiyar-lite)

`pytorch` · `multi-gpu` · `pretraining`

---

### [HueCast](https://github.com/Zephyr4772) — air quality monitoring with an LLM assistant

Real-time AQI for any city, pollutant breakdowns, and a Gemini-backed chat that answers questions in context of the current reading.

`typescript` · `react` · `gemini`

---

## Published datasets

- **[Indian_Supreme_Court_Judgments](https://huggingface.co/datasets/sinhal/Indian_Supreme_Court_Judgments)** — 41,876 judgments, converted from raw court PDFs to machine-readable text
- **[indian-pharma-dataset](https://huggingface.co/datasets/sinhal/indian-pharma-dataset-2026-augast)** — 120,000+ Indian drug records: pricing, salt composition, manufacturer, supplier, interactions

---

## Currently

Working on **Devanagari text rendering in diffusion models** — training an SD1.5 LoRA to generate legible Hindi script, which turns out to be hard for reasons that are specific and interesting: CLIP's tokenizer has no Devanagari coverage, and the VAE's 8× downsampling puts a hard floor on how small a legible glyph can be.

---

## Reach me

[Email](mailto:sinhalz4772@gmail.com) · [LinkedIn](https://www.linkedin.com/in/sinhaljjoshi) · [Hugging Face](https://huggingface.co/sinhal)
