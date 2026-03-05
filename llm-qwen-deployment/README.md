# Qwen Moral Story Serving Pipeline

High-throughput inference deployment for a children's moral story generation service using `Qwen/Qwen2.5-1.5B-Instruct`. Two pipelines are benchmarked side-by-side at 100K queries across 2 RunPod pods (data parallelism).

---

## Goal

Extend the optimizations explored in `LLM_inference_from_scratch.ipynb` into a production-grade serving pipeline and quantify the improvement that an industry-standard framework (vLLM) provides over a custom-built server with the same underlying optimizations.

---

## Architecture

```
Locust (100 users, local machine)
        │
        ├──── random round-robin ────┐
        │                            │
        ▼                            ▼
   Pod 1 (RunPod)              Pod 2 (RunPod)
   ┌──────────────┐            ┌──────────────┐
   │ Run A:       │            │ Run A:       │
   │ custom_      │            │ custom_      │
   │ server.py    │            │ server.py    │
   ├──────────────┤            ├──────────────┤
   │ Run B:       │            │ Run B:       │
   │ vLLM server  │            │ vLLM server  │
   └──────────────┘            └──────────────┘
        │                            │
        └────────────┬───────────────┘
                     ▼
             compare_results.py
             (side-by-side report)
```

Both pods serve independently — this is **data parallelism**: requests are distributed across pods, each holding a full model copy.

---

## Inference Pipelines

### Run A — Custom Pipeline (`custom_server.py`)

Built from scratch, extending the notebook experiments. Served via FastAPI + uvicorn.
Implements four progressive levels — each one closes the gap with vLLM.

| Level | Optimization | Implementation |
|---|---|---|
| 1 | Async queue + batch dispatch | `asyncio.Queue` → `asyncio.Future`; Level 1 was the starting point |
| 2 | **Continuous batching** | Manual `model.forward()` loop replaces `model.generate()`. Finished sequences are evicted mid-batch and new requests slot in immediately — no head-of-line blocking. |
| 3 | **KV cache slot pool** | Pre-allocated slot counter (`MAX_BATCH_SIZE`) limits active concurrency; excess requests queue in the bridge until a slot frees (natural backpressure, no OOM risk). |
| 4 | **Custom speculative decoding** | Draft model (`Qwen2.5-0.5B`) generates `K+1` tokens per slot; main model verifies `K` in **one batched forward pass** with left-padded KV caches. Accept/reject greedily per slot. Both KV caches updated consistently. |
| — | INT4 NF4 Quantization | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` |
| — | Flash Attention (SDPA) | `attn_implementation="sdpa"` on both main and draft models |

> **Why Level 4 bypasses HuggingFace's batch_size=1 restriction:**
> HuggingFace's `assisted_decoding` (called via `generate(assistant_model=...)`) only supports
> `batch_size=1` because it is a **convenience wrapper**, not the algorithm itself.
> By calling `model.forward()` directly and implementing the accept/reject loop ourselves,
> we remove that restriction entirely. Any batch size works.
>
> This was the key learning from smoke tests 1 & 2: the earlier `model.generate()` approach
> forced a tradeoff between speculative decoding and batching. The Level 2–4 implementation
> eliminates that tradeoff.

### Run B — vLLM Pipeline

Industry-standard serving. Start with RunPod's pre-built vLLM image.

| Optimization | vLLM handles automatically |
|---|---|
| PagedAttention | Block-based KV cache, eliminates memory fragmentation |
| Continuous Batching | Requests join/leave mid-batch — no head-of-line blocking |
| Flash Attention | Enabled by default on compatible hardware |
| OpenAI-compatible API | `/v1/chat/completions` — same endpoint as custom server |

---

## Repository Structure

```
llm-qwen-deployment/
├── custom_server.py       # Run A: FastAPI server with all custom optimizations
├── locustfile.py          # Load test (works for both pipelines via RUN_TAG)
├── gpu_monitor.py         # GPU utilization collector (runs alongside Locust)
├── llm_as_eval.py         # LLM-as-a-judge quality scorer
├── compare_results.py     # Side-by-side comparison report generator
├── generate_prompts.py    # 1,500 synthetic children's story prompts
├── eval_data.json         # 30 prompts + reference stories for quality eval
├── requirements.txt       # Dependencies
├── .env.example           # Environment variable template
└── results/
    ├── request_metrics_custom.csv
    ├── request_metrics_vllm.csv
    ├── gpu_metrics_custom.csv
    ├── gpu_metrics_vllm.csv
    ├── eval_scores_custom.csv
    └── eval_scores_vllm.csv
```

---

## Setup

### 1. Local machine

```bash
cd llm-qwen-deployment
python -m venv .venv && source .venv/bin/activate
pip install openai requests locust python-dotenv
cp .env.example .env   # fill in POD_1_URL, POD_2_URL, judge credentials
```

### 2. RunPod pods

Create two pods (RTX 4000 Ada or equivalent) using **RunPod's PyTorch image** (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` or similar). SSH into each and run:

```bash
pip install fastapi "uvicorn[standard]" transformers accelerate bitsandbytes pynvml torch
```

Then upload `custom_server.py` to each pod (e.g. via `scp` or paste into the RunPod terminal).

---

## Run A — Custom Pipeline

### Step 1 — Upload and start server on both pods

From your Mac (run once per pod, substituting SSH port and IP):

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    llm-qwen-deployment/custom_server.py \
    root@<POD_IP>:/root/custom_server.py
```

SSH into each pod and run:

```bash
MAX_BATCH_SIZE=16 NUM_SPECULATIVE_TOKENS=5 PORT=8888 \
    nohup python custom_server.py > server.log 2>&1 &
```

> Port 8888 is the default exposed port on RunPod pods.
> Both `_main_model` (INT4) and `_draft_model` (FP16) are loaded on startup.
> Expect ~2–3 minutes for model downloads + quantization.

Verify it's up (replace with your pod URL):

```bash
curl https://<POD_ID>-8888.proxy.runpod.net/health
```

### Step 2 — Run load test (Terminal 1, local machine)

```bash
cd llm-qwen-deployment

RUN_TAG=custom \
POD_1_URL=https://<POD1>-8888.proxy.runpod.net \
POD_2_URL=https://<POD2>-8888.proxy.runpod.net \
locust -f locustfile.py --headless \
    --users 50 \
    --spawn-rate 5 \
    --run-time 10h \
    --host https://<POD1>-8888.proxy.runpod.net \
    --csv results/locust_custom_100k
```

### Step 3 — Collect GPU metrics (Terminal 2, local machine)

```bash
cd llm-qwen-deployment

python gpu_monitor.py \
    --run-tag custom \
    --pod-url https://<POD1>-8000.proxy.runpod.net \
               https://<POD2>-8000.proxy.runpod.net \
    --interval 10
```

### Step 4 — Quality evaluation (after load test completes)

```bash
python llm_as_eval.py
cp results/eval_scores.csv results/eval_scores_custom.csv
```

---

## Run B — vLLM Pipeline

### Step 1 — Install vLLM and start on both pods

SSH into each pod (same pods as Run A — stop the custom server first):

```bash
# Stop the custom server
pkill -f custom_server.py

# Install vLLM
pip install vllm

# Start the vLLM OpenAI-compatible server
nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype bfloat16 \
    --max-model-len 662 \
    --gpu-memory-utilization 0.85 \
    --port 8888 > vllm.log 2>&1 &
```

Verify it's up:

```bash
curl https://<POD_ID>-8888.proxy.runpod.net/health
```

### Step 2 — Run load test (Terminal 1, local machine)

```bash
cd llm-qwen-deployment

RUN_TAG=vllm \
POD_1_URL=https://<POD1>-8888.proxy.runpod.net \
POD_2_URL=https://<POD2>-8888.proxy.runpod.net \
locust -f locustfile.py --headless \
    --users 50 \
    --spawn-rate 5 \
    --run-time 10h \
    --host https://<POD1>-8888.proxy.runpod.net \
    --csv results/locust_vllm_100k
```

### Step 3 — Collect GPU metrics (Terminal 2, local machine)

```bash
cd llm-qwen-deployment

python gpu_monitor.py \
    --run-tag vllm \
    --pod-url https://<POD1>-8000.proxy.runpod.net \
               https://<POD2>-8000.proxy.runpod.net \
    --interval 10
```

### Step 4 — Quality evaluation

```bash
python llm_as_eval.py
cp results/eval_scores.csv results/eval_scores_vllm.csv
```

---

## Comparison Report

Once both runs are complete:

```bash
python compare_results.py
```

Output is a markdown table covering:
- Total / successful / failed requests
- Avg, p50, p90, p99 response latency
- Throughput (req/s) and tokens/sec
- GPU utilization / VRAM
- LLM-as-a-judge scores across 4 quality dimensions

---

## LLM-as-a-Judge

`llm_as_eval.py` evaluates the **30 prompts in `eval_data.json`** against reference stories on four dimensions (each 0.0–1.0):

| Dimension | What it measures |
|---|---|
| Moral Clarity | Is the lesson clear and identifiable? |
| Age-Appropriateness | Is language and tone suitable for children 4–10? |
| Narrative Coherence | Does the story have a complete beginning/middle/end? |
| Relevance | Does the story match the requested character, moral, and setting? |

Judge model: configured via `JUDGE_API_KEY`, `JUDGE_API_BASE`, `JUDGE_MODEL` in `.env`.

---

## Inference Parallelism

This deployment uses **Data Parallelism (DP)**: each pod holds a complete copy of the model and handles independent requests. Locust distributes traffic randomly across both pod URLs.

Alternative strategies (not implemented here):
- **Tensor Parallelism (TP)**: splits individual weight matrices across multiple GPUs on the same machine — reduces per-GPU memory, enables larger models.
- **Pipeline Parallelism (PP)**: assigns different transformer layers to different GPUs — useful for very deep models that don't fit in a single GPU's memory.

For a 1.5B-parameter model on 20 GB VRAM, DP across 2 pods is the right strategy: the model fits comfortably on one GPU, so parallelizing the workload (not the model) is optimal.

---

## Experimental Results

### Run A — Smoke Test 1: Speculative Decoding + Batch Size 1

**Config:** `MAX_BATCH_SIZE=1`, `assistant_model=Qwen2.5-0.5B-Instruct`, 10 Locust users, single pod

| Metric | Result |
|---|---|
| Total requests | 104 |
| Failed requests | 2 (1.92%) — ReadTimeout after 120s |
| Avg response time | ~99,550 ms (~100 seconds) |
| Median (p50) latency | ~102,000 ms |
| Min latency | 13,213 ms |
| Max latency | 120,057 ms |
| Throughput | 0.09 req/s |

**Verdict:** Very slow. With `batch_size=1` enforced by speculative decoding, each of the 10 concurrent users had to wait for all previous requests to finish before being served — requests queued up serially. GPU was barely utilized.

**Root cause:** Speculative decoding is incompatible with dynamic batching (see note above). Running `MAX_BATCH_SIZE=1` means only one request is processed at a time, leaving 9 users waiting in the queue at any given moment.

---

### Run A — Smoke Test 2: Dynamic Batching (no speculative decoding)

**Config:** `MAX_BATCH_SIZE=8`, speculative decoding disabled, 10 Locust users, single pod

| Metric | Result |
|---|---|
| Total requests | 109 |
| Failed requests | 0 (0%) |
| Avg response time | ~27,250 ms (~27 seconds) |
| Median (p50) latency | ~27,000 ms |
| Min latency | 8,821 ms |
| Max latency | 32,435 ms |
| Throughput | 0.34–0.80 req/s |

**Comparison vs Smoke Test 1 (speculative decoding, batch_size=1):**

| Metric | Spec Decoding (batch=1) | Dynamic Batching (batch=8) | Improvement |
|---|---|---|---|
| Avg latency | ~99,550 ms | ~27,250 ms | **3.7× faster** |
| Failure rate | 1.92% | 0% | **Zero failures** |
| Max latency | 120,057 ms | 32,435 ms | **3.7× lower** |

**Key takeaway:** Switching from speculative decoding (forced batch=1) to dynamic batching (batch=8) delivered a **3.7× reduction in latency** and eliminated all failures. The GPU is now processing 8 requests simultaneously per forward pass instead of 1, dramatically improving utilization under concurrent load.

---

## What the comparison proves

| Feature | `custom_server.py` | vLLM |
|---|---|---|
| Async queue + batch dispatch | Yes (Level 1) | Yes |
| Continuous batching | Yes (Level 2) | Yes |
| KV cache slot reuse | Yes (Level 3) | Yes (PagedAttention) |
| Speculative decoding with batches | Yes (Level 4) | Yes |
| Memory fragmentation eliminated | No (contiguous tensors) | Yes (paged blocks) |
| Implementation language | Python | C++ / CUDA kernels |

The remaining gap — paged memory and CUDA-level optimisation — is precisely what justifies vLLM in production.

## Target Metrics

| Metric | Custom Pipeline | vLLM | Expected delta |
|---|---|---|---|
| Throughput (req/s) | baseline | target: 2–5× higher | vLLM PagedAttention + CUDA kernels |
| Avg latency (ms) | baseline | target: lower | vLLM contiguous KV, no fragmentation |
| p99 latency (ms) | baseline | target: lower | no head-of-line blocking |
| Tokens/sec | TBD | target: higher | Flash Attention CUDA kernel |
| Failure rate | < 1% | < 0.5% | — |
| Overall quality score | TBD | TBD | should be similar (same model) |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
POD_1_URL=https://<pod1-id>-8000.proxy.runpod.net
POD_2_URL=https://<pod2-id>-8000.proxy.runpod.net
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
JUDGE_API_KEY=...
JUDGE_API_BASE=https://api.openai.com/v1
JUDGE_MODEL=gpt-4o-mini
```
