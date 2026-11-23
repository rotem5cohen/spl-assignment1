# GPU Inference Server Simulator

A C++11 simulation of a multi-GPU inference server, featuring GPU memory management, KV-cache reuse, request scheduling, and resource-aware execution of generative and embedding workloads.

Developed as a collaborative Systems Programming course project at Ben-Gurion University.

## Overview

The project simulates how an inference server manages multiple AI models and incoming requests under limited GPU memory resources.

Rather than performing real GPU computation, the simulator models the infrastructure behind inference serving, including memory allocation, caching, scheduling, and request execution.

## Key Features

- **Multi-GPU Resource Management** — Tracks GPU memory utilization, block allocation, and reservations across multiple devices.
- **KV-Cache Management** — Reuses cached token prefixes to reduce redundant computation and supports cache eviction and reference counting.
- **Cache-Aware Scheduling** — Implements three scheduling strategies: Arrival Order, Shortest Job, and Longest Prefix.
- **Generative Inference Simulation** — Processes prompt tokens and generates scripted output while managing cached intermediate representations.
- **Embedding Workloads** — Computes deterministic token embeddings and aggregates them into output vectors.
- **Cross-GPU Cache Copying** — Supports copying cached blocks between GPUs.
- **Memory Safety** — Implements the Rule of Five, deep copying, move semantics, and polymorphic cloning.

## Technologies & Concepts

**Language:** C++11

**Core concepts:**
- Object-Oriented Programming and Polymorphism
- Dynamic Memory Management and RAII
- Copy and Move Semantics
- Rule of Five
- STL Containers and Algorithms
- Scheduling Algorithms
- Reference Counting
- Cache Management
- JSON Configuration Parsing
- GNU Make
- Valgrind

## Architecture

| Component | Responsibility |
|---|---|
| `Server` | Coordinates request admission, scheduling, GPU assignment, and execution cycles |
| `GPU` | Manages memory blocks, reservations, cache lookup, and eviction |
| `MemoryBlock` | Provides polymorphic memory storage for KV-cache and embedding workspace blocks |
| `Request` | Defines generative and embedding request behavior |
| `SchedulingPolicy` | Implements interchangeable request prioritization strategies |
| `Model` | Handles tokenization and deterministic synthetic embeddings |

## Building

Requirements:
- A C++11-compatible compiler (G++)
- GNU Make

```bash
make clean
make
```

## Running

The simulator reads its configuration from a JSON file:

```bash
./bin/server config.json
```

The configuration specifies the GPU resources, supported models, incoming requests, and scheduling policy.

## Testing & Memory Validation

The implementation was tested using multiple simulation scenarios covering caching, GPU selection, request processing, and copy/move semantics.

Memory correctness was checked using Valgrind:

```bash
valgrind --leak-check=full --show-reachable=yes ./bin/server config.json
```

The tested scenarios completed without reported memory leaks or invalid memory accesses.

## Project Context

This project was developed as part of the Systems Programming course at Ben-Gurion University.

It is an educational simulator, not a production inference engine. GPU execution and embeddings are simulated rather than implemented using CUDA or a real machine-learning runtime.

Developed collaboratively by two students.