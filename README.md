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

### [yolo26-laptop-benchmark](https://github.com/Zephyr4772/yolo26-laptop-benchmark) — YOLO26 vs YOLO11 across 8 runtimes

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

### [Dr.Data](https://github.com/Zephyr4772/Dr.Data) — can this dataset support this use case?

A pre-training audit for tabular data. You give it a CSV and a use case (target, goal, time and entity columns, columns not known at prediction time). It runs ~200 controlled experiments and tells you whether training is worth it, then how to train.

- **Label-shuffle test:** a retail return-prediction target scored ROC-AUC 0.48 against 0.47–0.53 on shuffled labels. There was no learnable signal, so the verdict was "don't train, collect new information".
- **Exact-formula leak search:** found `Profit = Sales − Cost` in 99.4% of rows and flagged the 106 rows that break it as data errors.
- **Validation realism:** the same model scored R² 0.87 on a random split and 0.75 on a past-to-future split.
- **Ground-truth check:** on a deliberately sabotaged copy of a clean dataset (score 97 → 60), it recovered every planted defect, estimating 6.5% label noise against ~6% planted.

It then recommends a model (6-family bake-off with the one-standard-error rule), a decision threshold, loss curves to expect, a training-time estimate and scikit-learn starter code.

`scikit-learn` · `fastapi` · `data-quality` · `evaluation`

---

### [factory-sim](https://github.com/Zephyr4772/factory-sim) — discrete-event pipeline simulator in C++20

Design a production line, service desk or software pipeline from sources, machines, conveyors, batchers, quality gates and sinks, then run it and watch buffers fill and machines starve, block and break down. It finds the bottleneck and measures which fix gives the most throughput.

- **Engine:** min-heap event queue, integer-microsecond clock, bounded buffers with backpressure, MTBF/MTTR failures, deadlock detection, bit-reproducible runs.
- **Factory Studio:** a browser drag-and-drop editor with animated replay, what-if analysis and 20-seed confidence intervals.
- **Audit report:** a 0–100 health score with graded findings, as a multi-page PDF.

`c++20` · `cmake` · `simulation`

---

## Writing

- **[A 2019 classifier ties a 2026 decision model on agent guardrails](https://zephyr4772.github.io/2019-classifier-ties-2026-decision-model/)**: zero-shot, the open decision model Laya (52.5%) is statistically tied with BART-large-MNLI from 2019 (49.2%) on 244 agent-guardrail decisions, and neither beats always giving the same answer (67.6%).

- **[On a laptop, the YOLO version is the smallest factor in how fast it runs](https://zephyr4772.github.io/yolo26-laptop-benchmark/)**: YOLO26 vs YOLO11 changes latency by about 5%, the runtime by 5.4×, and Windows' power mode by up to 7.8×, which nearly made me publish that YOLO26 was 50–90% faster.

More at **[zephyr4772.github.io](https://zephyr4772.github.io)**.

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

[Email](mailto:sinhalz4772@gmail.com) · [LinkedIn](https://www.linkedin.com/in/sinhaljjoshi) · [Hugging Face](https://huggingface.co/sinhal) · [Blog](https://zephyr4772.github.io)
