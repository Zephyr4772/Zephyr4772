# Sinhal Joshi

**I build and evaluate LLM agent systems** — retrieval pipelines, agent tooling, and the evals that tell you whether any of it actually works.

Most of what I publish falls into one of two buckets: systems that retrieve and reason over a real corpus, and harnesses that measure where a model fails. The second is the part I find more interesting, and it's where my strongest work is.

---

## Open-source contributions

**14 merged PRs** in [Ultralytics](https://github.com/ultralytics/ultralytics), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [evalscope](https://github.com/modelscope/evalscope), [kornia](https://github.com/kornia/kornia), [doctr](https://github.com/mindee/doctr), [jsonargparse](https://github.com/mauvilsa/jsonargparse) and [jc](https://github.com/kellyjonbrazil/jc), mostly finding evaluation and inference bugs that fail silently.

- **[evalscope #2085](https://github.com/modelscope/evalscope/pull/2085): added the BBEH (BIG-Bench Extra Hard) benchmark.**
  - All 23 tasks, with DeepMind's official answer matching ported as-is.
  - BBEH Mini, and the paper's headline harmonic-mean score. The formula reproduces the reported number for all 13 models in the paper.
  - On a real model run, all 460 BBEH Mini answers scored exactly as with DeepMind's unmodified scorer.
- **[kornia #6816](https://github.com/kornia/kornia/pull/6816): made `AugmentationSequential` 22–37% faster on the GPU for common image pipelines.**
  - The container built every child's transformation matrix on each call, even when nothing read it. It now builds them only when `.transform_matrix` is read.
  - Outputs, matrices, boxes, keypoints and masks stay bitwise identical. The speed-up is the maintainer's own measurement on an RTX 4090, at batch 16 to 64.
- **[kornia #6899](https://github.com/kornia/kornia/pull/6899): made `crop_by_indices` crop the same under `torch.export`, ONNX and `torch.compile` as in eager mode.**
  - Graph capture resampled any box that wasn't exactly the output size with `align_corners=True`, up to 0.98 off the eager crop. It now uses the same gather as `torch.compile`.
  - Checked on torch 2.5 through 2.15: the ONNX difference went from 0.25/0.23/0.99 to 1e-7/0/0 (bilinear/bicubic/nearest).
- **[doctr #2232](https://github.com/mindee/doctr/pull/2232): restored 8.9 and 12.0 points of word accuracy to the MASTER and SAR text recognizers.**
  - Since May, each had lost one ReLU from its backbone, because a default activation module was shared between layers. The bug shipped in the v1.1.0 release.
  - Found in real use (MASTER garbled an invoice), bisected to the commit that introduced it, and measured on IIIT5K.
- **[evalscope #2084](https://github.com/modelscope/evalscope/pull/2084):** local-model evals failed on every sample once transformers 5.23 started requiring chat messages as plain dicts. Caught on the day of that release.
- **[ultralytics #27175](https://github.com/ultralytics/ultralytics/pull/27175):** with CUDA 12 PyTorch, auto-install pulled a CUDA 13 `onnxruntime-gpu`, so every ONNX predict, val and benchmark silently ran on the CPU. Reproduced on an RTX 4060.
- **[ultralytics #26624](https://github.com/ultralytics/ultralytics/pull/26624):** TensorRT 11 export let the `nvidia-modelopt` auto-install replace CUDA PyTorch with a CPU-only wheel. The fix pins the installed torch through the existing `constrain=` mechanism. Reported in [#26623](https://github.com/ultralytics/ultralytics/issues/26623).
- **lmms-eval, 5 merged:**
  - [#1667](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1667): checked every task config against the Hugging Face Hub and fixed 18 tasks pointing at configs or splits that don't exist.
  - [#1671](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1671): chat models were shown every image twice on STARE, MMT, Geometry3K, MathCanvas and PhysBench.
  - [#1668](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1668): HF chat models failed every request under transformers 5.
  - [#1666](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1666): the generic HF chat model loaded an Auto class with no `generate()`.
  - [#1670](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1670): six vision models crashed on a single GPU.
- **[jsonargparse #1035](https://github.com/mauvilsa/jsonargparse/pull/1035):** config files were saved and read with the locale encoding, so on Windows `save()` crashed on non-ASCII values and UTF-8 configs were misread.
- **[jc #918](https://github.com/kellyjonbrazil/jc/pull/918):** on Windows, `jc --kv < file` printed `{}`. The `os.system('')` call that turns on terminal colors started a cmd.exe that inherited jc's input and read a redirected file to the end. Reported in [#917](https://github.com/kellyjonbrazil/jc/issues/917).

**In review:**
- **Numbers in Indian-language AI** ([gujnorm](https://github.com/Zephyr4772/gujnorm)): fixes from testing how Sarvam's TTS, AI4Bharat's tools and IndicTrans2 handle written Gujarati and Hindi numbers.
  - [sarvamai/indic-numtowords #5](https://github.com/sarvamai/indic-numtowords/pull/5) and [AI4Bharat/indic-numtowords #16](https://github.com/AI4Bharat/indic-numtowords/pull/16): a misspelled Hindi "सवा" in 9 entries, Gujarati crore written in Punjabi script, and the missing Gujarati variations.
  - [IndicTransToolkit #32](https://github.com/VarunGumma/IndicTransToolkit/pull/32): Indian-grouped ₹ amounts were split or changed in translation ("₹1 crore" became "₹1,000"). Split numbers drop from 26 to 3 per model, with no new errors.
  - [sarvamai/skills #47](https://github.com/sarvamai/skills/pull/47): documents that Gujarati numbers need normalizing before Bulbul TTS.
- **[evalscope #2193](https://github.com/modelscope/evalscope/pull/2193):** the MMLU-Indic benchmark, Sarvam's MMLU in 10 Indian languages in native and romanized script (21 configs, 293k questions).
- **[evalscope #2209](https://github.com/modelscope/evalscope/pull/2209):** local-model evals with `temperature: 0` crashed on every sample for Qwen and Llama instruct models; temperature 0 now means greedy decoding.
- **[ms-swift #10591](https://github.com/modelscope/ms-swift/pull/10591):** SmolVLM and SmolVLM2 support, and a fix to the Idefics3 template so its encodings match the Hugging Face processor token for token.
- **Data tools on Windows:** bugs found by running the tools on Windows, where their CI didn't reach.
  - [csvkit #1395](https://github.com/wireservice/csvkit/pull/1395): csvkit couldn't read its own output (`csvcut a.csv | csvsort` failed on "é"), and `--add-bom` wrote a UTF-8 BOM before cp1252 text.
  - [datacompy #619](https://github.com/capitalone/datacompy/pull/619) and [markdown-it-py #486](https://github.com/executablebooks/markdown-it-py/pull/486): redirecting the report or HTML to a file crashed on characters outside cp1252.
  - [jupytext #1719](https://github.com/jupytext/jupytext/pull/1719): `--paired-paths` always exited with 1, so editor extensions couldn't tell success from failure.

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

- **[Half past nine, ₹1.5 lakh, ₹1 crore: where Indian-language AI gets numbers wrong](https://zephyr4772.github.io/gujarati-numbers/)**: Sarvam's Bulbul voices misread half the Gujarati sentences that contain a number (59/55/56 of 108); a 0.13 ms text normalizer brings that to 0/2/1, confirmed by a blind listening test (0/20 → 19/20, p = 0.00008). Also: Sarvam's spoken-form API turning 9:30 into 10:15, and IndicTrans2 turning ₹1 crore into ₹1,000.

- **[Eval harnesses fail in two ways, and the quiet one is worse](https://zephyr4772.github.io/eval-harness-quiet-failures/)**: bugs I fixed in three LLM and VLM evaluation harnesses where nothing crashed: a math metric that scored `7 = 7` as wrong on Windows, saved results that vanished into a hidden NTFS stream, prompts that showed every image twice, and 32 benchmark tasks whose Hub data no longer exists.

- **[Ultralytics' auto-install quietly took my GPU away, twice](https://zephyr4772.github.io/ultralytics-autoinstall-gpu/)**: once by installing a CUDA 13 ONNX Runtime next to CUDA 12 PyTorch, so models ran on the CPU about 3× slower, and once by swapping CUDA PyTorch for a CPU-only build in the middle of a TensorRT export.

- **[No model can fix data that doesn't fit the question](https://zephyr4772.github.io/same-csv-different-verdicts/)**: with Dr.Data, one retail table scores 60/100 for predicting profit and 30/100 for predicting returns (the real labels do worse than shuffled ones), and a deliberately sabotaged medical dataset gives up every planted defect.

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
