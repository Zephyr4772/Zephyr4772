# Sinhal Joshi

**I build and evaluate LLM agent systems** — retrieval pipelines, agent tooling, and the evals that tell you whether any of it actually works.

Most of what I publish falls into one of two buckets: systems that retrieve and reason over a real corpus, and harnesses that measure where a model fails. The second is the part I find more interesting, and it's where my strongest work is.

---

## Open-source contributions

- **[ultralytics/ultralytics #26624](https://github.com/ultralytics/ultralytics/pull/26624)** (merged): fixed TensorRT 11 FP16/INT8 export silently replacing a CUDA PyTorch build. The `nvidia-modelopt` auto-install let the resolver swap `torch 2.5.1+cu121` for a CPU-only PyPI wheel and half-uninstall `onnx`. The fix pins the installed torch through the existing `constrain=` mechanism. I reproduced it with dry runs of the real install path, and it was merged by the maintainers. Reported in [#26623](https://github.com/ultralytics/ultralytics/issues/26623).

---

## Selected work

### [laya-zeroshot-agent-guard-eval](https://github.com/Zephyr4772/laya-zeroshot-agent-guard-eval) — can a small model act as an agent step-guard?

61 hand-written agent transcripts × 4 decisions = **244 judgments** across 3 checkpoints.

What it found:

| Failure mode | Result |
|---|---|
| Believes the agent's `"done"` over tool output showing failures | **0 / 6** |
| Never escalates destructive commands (`DELETE FROM users;` → *continue*) | **0 / 8** |
| Continue/stop score is inverted — flipping it would score better | **AUROC 0.32** |

Reported with bootstrapped confidence intervals and a majority-class baseline that beats all three checkpoints. Follow-ups:
- **Truncation:** a probe of the `laya` package's list-vs-dict truncation, prompted by a [community discussion](https://huggingface.co/convaiinnovations/laya/discussions/15).
- **Julia-1:** a head-to-head with [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), which never chose to escalate once in 61 transcripts.

The scenarios are an open benchmark: **[agent-step-guard-bench](https://huggingface.co/datasets/sinhal/agent-step-guard-bench)**

`python` · `transformers` · `evaluation`

---

### [indian-sc-outcome-bench](https://github.com/Zephyr4772/indian-sc-outcome-bench) — reasoning or recall on Indian Supreme Court judgments?

380 judgments with cross-checked labels: predict allowed vs dismissed from the facts and arguments. It has **matched recent (2023–24) and old (1995–2005) halves** plus a name-only recall probe, to separate reasoning from memorisation.

- **Five open 7–8B models are at chance:** 49–55% balanced accuracy.
- **No memorisation is visible.**
- **Mistral-7B declines to answer about post-cutoff cases from the date alone:** 160 of 190 recent cases, 0 of 190 old ones.

Run on Kaggle 2× T4 with vLLM. Building it also surfaced and documented a **4.5% metadata/judgment-text mismatch** in the source dataset.

`vllm` · `kaggle` · `evaluation` · `legal-nlp`

---

### [yolo26-laptop-benchmark](https://github.com/Zephyr4772/yolo26-laptop-benchmark) — YOLO26 vs YOLO11 on a gaming laptop

8 runtimes (PyTorch, ONNX Runtime, OpenVINO FP32/INT8, CUDA FP32/FP16, TensorRT FP32/FP16), full COCO val2017 accuracy, held-out INT8 calibration, and interleaved timing.

- **Speed is a tie,** and YOLO26 is +0.6–0.8 mAP more accurate.
- **The bigger finding:** Windows' *Best power efficiency* mode moves Python processes onto slower cores a few seconds after start-up, which made every runtime **2–8× slower**. It comes with a 10-second repro script.
- **TensorRT 11 FP16 engines are built directly,** as strongly typed networks, without the Ultralytics exporter.

`onnxruntime` · `openvino` · `tensorrt` · `benchmarking`

---

### [Lex.ai](https://github.com/Zephyr4772) — RAG over 41,839 Indian Supreme Court judgments

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

## Published datasets

- **[Indian_Supreme_Court_Judgments](https://huggingface.co/datasets/sinhal/Indian_Supreme_Court_Judgments)** — 41,839 judgments converted from raw court PDFs to machine-readable text, with a measured known-issues section
- **[indian-sc-outcome-bench](https://huggingface.co/datasets/sinhal/indian-sc-outcome-bench)** — 380 judgments with cross-checked outcome labels, split recent vs old, with results for five open models
- **[agent-step-guard-bench](https://huggingface.co/datasets/sinhal/agent-step-guard-bench)** — 61 coding-agent transcripts for testing step-level guard models, with baseline predictions
- **[indian-pharma-dataset](https://huggingface.co/datasets/sinhal/indian-pharma-dataset-2026-augast)** — 120,000+ Indian drug records: pricing, salt composition, manufacturer, supplier, interactions

---

## Currently

Working on **Devanagari text rendering in diffusion models** — training an SD1.5 LoRA to generate legible Hindi script, which turns out to be hard for reasons that are specific and interesting: CLIP's tokenizer has no Devanagari coverage, and the VAE's 8× downsampling puts a hard floor on how small a legible glyph can be.

---

## Reach me

[Email](mailto:sinhalz4772@gmail.com) · [LinkedIn](https://www.linkedin.com/in/sinhaljjoshi) · [Hugging Face](https://huggingface.co/sinhal)
