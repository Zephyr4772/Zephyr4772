## Sinhal Joshi

I build and evaluate LLM agent systems — retrieval pipelines, agent tooling, and
the evals that tell you whether any of it actually works.

Most of what I publish falls into one of two buckets: **systems that retrieve and
reason over a real corpus**, and **harnesses that measure where a model fails**.
The second is the part I find more interesting, and it's where my strongest work is.

---

### Selected work

**[laya-zeroshot-agent-guard-eval](https://github.com/Zephyr4772/laya-zeroshot-agent-guard-eval)** — Can a small model act as an agent step-guard?
> 61 hand-written agent transcripts × 4 decisions = 244 judgments across 3 checkpoints.
> Findings: it believes an agent's "done" over the tool output showing failures (0/6),
> never escalates destructive commands (`DELETE FROM users;` → *continue*, 0/8), and its
> continue/stop score is **inverted** (AUROC 0.32 — flipping it would score better).
> Reported with bootstrapped CIs and a majority-class baseline that beats all three checkpoints.
>
> `python` · `transformers` · `evaluation`

**[Lex.ai](https://github.com/Zephyr4772/lex-ai)** — RAG over 41,876 Indian Supreme Court judgments.
> Local ChromaDB index, `nomic-embed-text` embeddings via Ollama, streaming analysis from
> Gemini 2.5-flash with automatic fallback to a local model. Retrieval stays on-device;
> only retrieved passages reach the LLM. Every claim renders as a traceable case card.
>
> `fastapi` · `chromadb` · `ollama` · `react`

**[ShipScript](https://github.com/Zephyr4772/ShipScript)** — An MCP server that documents your repo for you.
> Analyzes a GitHub repo, detects the stack, screenshots the live deployment, generates
> charts, and drafts a README plus launch posts. Exposed as MCP tools, with a Gradio UI.
>
> `mcp` · `litellm` · `playwright` · `gradio`

**[HueCast](https://github.com/Zephyr4772/HueCast)** — Air quality monitoring with an LLM assistant.
> Real-time AQI for any city, pollutant breakdowns, and a Gemini-backed chat that answers
> questions in context of the current reading.
>
> `typescript` · `react` · `gemini`

---

### Currently

Working on Devanagari text rendering in diffusion models — training a SD1.5 LoRA to
generate legible Hindi script, which turns out to be hard for reasons that are
specific and interesting: CLIP's tokenizer has no Devanagari coverage, and the VAE's
8× downsampling puts a hard floor on how small a legible glyph can be.

### Reach me

[![Email](https://img.shields.io/badge/email-sinhalz4772%40gmail.com-informational?style=flat-square)](mailto:sinhalz4772@gmail.com)
