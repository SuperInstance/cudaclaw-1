# cudaclaw

Rust+CUDA framework for GPU-resident agent execution. Forked from [SuperInstance/cudaclaw](https://github.com/SuperInstance/cudaclaw) and extended for the Cocapn fleet.

## Core Architecture

### Persistent GPU Kernel (`kernels/executor.cu`)
- Lock-free SPSC queue in unified memory (<5us dispatch-to-execution)
- Warp-parallel command dispatch (32 threads per warp)
- SmartCRDT integration — atomicCAS conflict resolution without leaving GPU
- Zero `cudaDeviceSynchronize()` in hot path

### Cell Agents (`src/gpu_cell_agent/`)
- Each agent = GPU-compatible struct (`#[repr(C)]`, 48 bytes)
- AoS on host → SoA on GPU for coalesced memory access
- States: Idle, Executing, Blocked, Completed, Error, Migrating
- Fiber affinity: each agent assigned to optimized kernel config

### Muscle Fibers (`src/gpu_cell_agent/muscle_fiber.rs`)
- Named kernel configs: `cell_update`, `crdt_merge`, `formula_eval`, `batch_process`, `idle_poll`
- Block size, shared memory budget, register budget per fiber
- ML feedback reassigns agents to fibers dynamically

### Ramify Engine (`src/ramify/`)
- **PTX Branching**: runtime kernel specialization based on access patterns
- **Shared Memory Bridges**: ~5-cycle shmem vs ~400-cycle global memory
- **Resource Exhaustion**: SM rebalancing when agents exhaust registers/shared mem
- **NVRTC Compiler**: CUDA C++ → PTX in 10-50ms without nvcc

### DNA (`src/dna.rs`)
- `.claw-dna` files: complete instance blueprint (JSON)
- Hardware fingerprint (compute capability, SM count, L2 cache, bandwidth)
- Constraint-theory mappings (safe bounds from physics)
- PTX muscle fibers (kernel configs + source code)
- Resource exhaustion metrics (feedback signal for mutation)

### ML Feedback Loop (`src/ml_feedback/`)
- ExecutionLog → SuccessAnalyzer → DnaMutator → Constraint DNA
- Mutation strategies: Tighten, Relax, Specialize
- Pattern detection across execution histories

### SmartCRDT (`kernels/smartcrdt.cuh`)
- GPU-resident CRDT with Lamport clocks
- atomicCAS-based LWW conflict resolution
- Warp-parallel conflict resolution (no global memory round-trips)

## Lucineer Extensions

Planned extensions for Cocapn fleet integration:
- Fleet DNA: multi-agent GPU coordination templates
- Inference Fibers: INT4/INT8 systolic array kernels for edge inference
- Trust Propagation: parallel fleet-wide trust computation on GPU
- Sensor Fusion: GPU-accelerated Bayesian fusion across agent array
- Yield Simulation: Monte Carlo die yield across 10K+ dies simultaneously
- Fault Simulation: parallel stuck-at fault simulation across gate arrays

## Build

```bash
# Without CUDA (pure Rust, simulated GPU)
cargo build

# With CUDA
cargo build --features cuda
```

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                  Rust Host                            │
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Dispatcher│  │ Agent Mgr │  │ Ramify Engine    │  │
│  └────┬─────┘  └────┬──────┘  └────────┬─────────┘  │
│       │              │                   │            │
│  ┌────▼──────────────▼───────────────────▼────────┐  │
│  │         Unified Memory (CommandQueue)          │  │
│  └─────────────────────┬─────────────────────────┘  │
└────────────────────────┼────────────────────────────┘
                         │ PCIe / NVLink
┌────────────────────────┼────────────────────────────┐
│                  GPU Device                         │
│  ┌─────────────────────▼─────────────────────────┐  │
│  │         Persistent Kernel (executor.cu)       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐  │  │
│  │  │ Warp 0   │  │ SmartCRDT│  │ Muscle     │  │  │
│  │  │ Dispatch │  │ Engine   │  │ Fibers     │  │  │
│  │  └──────────┘  └──────────┘  └────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## License

MIT — DiGennaro et al. (SuperInstance & Lucineer)
