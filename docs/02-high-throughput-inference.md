# ⚡ Guide 02: High-Throughput Model Serving & Dynamic Batching

## 1. The Serving Latency vs. Throughput Trade-off

When deploying models in real-time microservices, individual incoming requests arrive concurrently at random intervals. Processing requests individually ($B=1$) results in low GPU compute utilization ($<15\%$) and high cost. Conversely, large static batches introduce excessive queue waiting latency.

**Dynamic Batching** dynamically coalesces incoming client requests over a sliding time window $\Delta t$ (e.g. $5\text{ms}$):
* If the batch reaches `max_batch_size` (e.g. 32), dispatch immediately.
* If `max_queue_delay_microseconds` expires, dispatch whatever requests are currently queued.

```mermaid
flowchart LR
    subgraph Clients[" Concurrent Client Requests "]
        R1["Req 1"]
        R2["Req 2"]
        R3["Req 3"]
    end

    subgraph Batcher[" Triton / Ray Dynamic Batcher "]
        Q["Priority Request Queue<br/>(Max Delay = 4ms, Max Batch = 32)"]
    end

    subgraph GPU[" TensorRT / ONNX Runtime Execution "]
        TensorRT["Fused Kernel Execution<br/>(Single Batched Matrix Multiplication)"]
    end

    R1 --> Q
    R2 --> Q
    R3 --> Q
    Q --> TensorRT

    style Clients fill:#1e2327,stroke:#4c72b0,stroke-width:1.5px,color:#ffffff
    style Batcher fill:#1e2327,stroke:#f59e0b,stroke-width:1.5px,color:#ffffff
    style GPU fill:#1e2327,stroke:#22c55e,stroke-width:1.5px,color:#ffffff
```

---

## 2. Kernel Fusion & Graph Compilation (TensorRT / ONNX Runtime)

PyTorch models execute operations as separate discrete GPU kernels, incurring memory bandwidth round-trips to global HBM memory:
* `LayerNorm` $\rightarrow$ Write HBM $\rightarrow$ Read HBM $\rightarrow$ `GELU` $\rightarrow$ Write HBM $\rightarrow$ Read HBM $\rightarrow$ `Linear`.

**Graph Compilers (TensorRT / torch.compile)** fuse adjacent operations into single compound GPU kernels, executing directly inside fast on-chip SRAM cache ($L1$/Shared Memory), reducing inference latency by **$2\times$ to $4\times$**.
