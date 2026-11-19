# Continuous Streaming Work Coordinator Architecture
*Inspired by Stockfish Fishtest, Lichess Fishnet, and Actor-based Double-Buffering*

## 1. The Core Problem with Batch Slices
Traditional batching creates two critical flaws:
1. **Idle Bubbles (Network Latency Starvation)**: Workers finish batch $N$, stop computing, upload results, request batch $N+1$, download positions, and only then resume search. This burns 5–20% of compute time in idle wait.
2. **Fragmented Queue Islands**: Hardcoded offsets (e.g. Worker A takes 0..500, Worker B takes 1200..1800, Worker C takes 2600..3600) create gaps. The highest-value training data (the disagreement head) gets skipped while workers relabel lower-value tail positions.

---

## 2. Double-Buffered Continuous Streaming Protocol

### Worker State: Dual-Slot Pipeline (Queue Depth = 2)
Each worker (local process, Docker sandbox, or Modal container) maintains two memory/file slots:
```
+-----------------------------------------------------------+
| Worker Compute Engine (100% CPU / GPU)                    |
| Currently Searching: Slot A (Chunk N)                     |
+-----------------------------------------------------------+
                              | (0 ms handoff upon finish)
                              v
+-----------------------------------------------------------+
| Staged Prefetch: Slot B (Chunk N+1)                       |
| Already downloaded, verified, and parsed in RAM          |
+-----------------------------------------------------------+
                              ^
                              | (Async Background I/O)
+-----------------------------------------------------------+
| Central Head Coordinator (HTTP / Atomic Shared Memory)    |
| Streams results of N-1 back, claims Chunk N+2 from HEAD    |
+-----------------------------------------------------------+
```

### Protocol Cycle:
1. **Start**: Worker requests initial **Slot A (Chunk N)** and **Slot B (Chunk N+1)**.
2. **Execution**: Worker begins processing Slot A.
3. **Instant Handoff (0 ms idle)**:
   - When Slot A finishes, the worker swaps active execution pointer immediately to Slot B.
   - No wait for network, no disk write locks blocking the search.
4. **Asynchronous Drain & Replenish**:
   - An independent I/O task flushes Slot A's results back to the central repository.
   - The coordinator marks Chunk N as `DONE`.
   - The I/O task requests the next lowest uncompleted chunk from the HEAD (Chunk N+2) and fills Slot A.
   - Slot A is now ready as the prefetch buffer before Slot B finishes.

---

## 3. Atomic Head Frontier Allocator

The Coordinator enforces a single monotonic invariant:
$$\text{Next Chunk} = \min \{ c \in [0, N_{\text{total}}) \mid c \notin \text{Completed} \land c \notin \text{InFlight} \}$$

* **No Disconnected Slices**: Slices are never partitioned by static ranges.
* **Lease Timeout / Deadlock Protection**: If a worker crashes or a container drops, its claimed chunks are returned to the head pool after a 10-minute timeout.
* **Zero Contention**: PC local workers take batches of 1–4 chunks; Cloud containers take batches of 8–16 chunks; Modal takes 100-container parallel waves. All waves are drawn strictly from the lowest missing indices.

---

## 4. Implementation Plan for Titanium

1. **Lightweight Head Coordinator Service** (built into `relabel_dashboard.py`):
   - Endpoints:
     - `GET /api/claim?worker_id=X&count=K`: Claims next $K$ chunks strictly from head, returns chunk IDs + binary byte payloads.
     - `POST /api/submit`: Submits completed lines, verifies checksum, marks done.
2. **Worker Client (`training/stream_worker.py`)**:
   - Spawns engine `--threads 1` processes.
   - Manages Slot A / Slot B double buffering with `asyncio` or background thread.
   - Compatible with local PC, Docker sandboxes, and Modal.
