# LLM Inference Optimization from Scratch

A comprehensive exploration of modern inference optimization techniques for Large Language Models, implemented from first principles using PyTorch.

##  Project Overview

This repository demonstrates 7 critical optimization techniques that power production LLM systems like GPT, Llama, and Mistral. Each phase builds upon the previous one, progressively reducing latency and memory footprint while maintaining model quality.

**Status:**  Work in Progress (Phases 1-5 complete, 6-7 coming soon)

---

## 📚 Optimization Phases

### ✅ Phase 1: KV Caching
**The foundation of efficient autoregressive decoding**

- Implements single-head and multi-layer attention with KV cache support
- Compares naive recomputation vs. cached approach across different sequence lengths
- **Key Result:** Up to **10x speedup** for long sequences by avoiding redundant key/value computations

### ✅ Phase 2: Peak GPU Utilization through Batching
**Maximizing hardware throughput**

- Explores batch scaling to saturate GPU compute (targeting peak FLOPS)
- Benchmarks both custom models and GPT-2 with varying batch sizes
- **Key Insight:** Batch processing amortizes memory bandwidth costs and dramatically improves tokens/sec

### ✅ Phase 3: Sliding Window Attention
**Memory-bounded long context handling**

- Implements attention with configurable window sizes (32, 64, 128, 256 tokens)
- Analyzes memory/speed trade-offs as sequence length grows
- **Key Trade-off:** Constant O(window_size) memory for KV cache vs. loss of long-range dependencies

### ✅ Phase 4: Flash Attention
**Kernel-level memory optimization**

- Integrates PyTorch's `scaled_dot_product_attention` with Flash/memory-efficient backends
- Compares FP32 naive attention vs. FP16 fused kernels
- **Hardware Note:** Demonstrates memory-efficient attention on Turing GPUs (T4); Flash Attention v2 requires Ampere+ (A100, RTX 3090)
- **Why both Phase 3 & 4?** Sliding window caps *context length*; Flash optimizes *computation* — orthogonal optimizations used together in production

### ✅ Phase 5: Paged Attention (vLLM-style)
**Dynamic memory allocation for batched inference**

- Custom block-based KV cache with fixed-size pages
- Enables memory sharing and reuse across sequences (prefix sharing, beam search)
- **Key Result:** ~4x memory savings vs. naive per-sequence allocation for variable-length batches

### 🔜 Phase 6: Speculative Decoding
**Latency reduction via draft models**

Coming soon: Using small draft models to predict multiple tokens, verified by the target model in parallel.

### 🔜 Phase 7: Production Deployment
**End-to-end serving pipeline**

Coming soon: Model quantization, continuous batching, and deployment with inference frameworks (vLLM/TGI/TensorRT-LLM).

---

## 🛠️ Technical Stack

- **Framework:** PyTorch 2.x with CUDA support
- **Models:** Custom transformer blocks, HuggingFace GPT-2
- **Hardware:** NVIDIA GPUs (tested on T4; Ampere+ recommended for full Flash Attention support)
- **Visualization:** Matplotlib for performance analysis

---

## 🎯 Learning Outcomes

By working through this notebook, you will understand:

1. **Why KV caching is mandatory** for any production LLM inference system
2. **How batching exploits GPU parallelism** to maximize throughput
3. **The memory/context trade-off** in windowed attention mechanisms
4. **Kernel fusion techniques** that reduce memory bandwidth bottlenecks
5. **Advanced memory management** strategies for multi-sequence batched serving

Each phase includes:
- ✅ Clean reference implementations
- ✅ Benchmarking code with time/memory profiling
- ✅ Comparative visualizations (speedup curves, memory usage)
- ✅ Detailed explanations of when and why each technique matters

---

## 📊 Sample Results

**Phase 1 (KV Cache):** 400 tokens → 2.5s (cached) vs. 25.3s (naive) = **10.1x speedup**

**Phase 3 (Sliding Window):** 2048 tokens → 512 MB (full) vs. 32 MB (w=128) = **16x memory reduction**

**Phase 5 (Paged Attention):** 5 variable-length sequences → 4.2 MB (paged pool) vs. 16.8 MB (naive pre-alloc) = **4x savings**

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd LLM-Inference-Optimisation

# Install dependencies
pip install torch torchvision transformers matplotlib

# Launch Jupyter
jupyter notebook LLM_inference_from_scratch.ipynb
```

**GPU Recommended:** While the code will run on CPU, performance benefits are only observable with CUDA-enabled GPUs.

---
