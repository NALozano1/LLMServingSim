# MoE Expert-Selection Traces (prefill routing)

Per-token, per-layer expert selections captured from real vLLM prefill (v0.18.1,
`enable_return_routed_experts=True`) across MoE LLMs. One array per model.

## Format
`traces/<org>_<model>/prefill_routing.npy` — `int16`, shape **`[num_tokens, num_layers, top_k]`**.
Each entry is the selected expert **id** (0-based) for that (token, layer, slot).
The token axis is one contiguous prefill stream (~64k tokens) over the shared prompt set.

```python
import numpy as np
a = np.load("traces/mistralai_Mixtral-8x7B-v0.1/prefill_routing.npy")  # (64635, 32, 2)
# per-layer expert load histogram:
import numpy as np
counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=a.max()+1), 0,
                             a.reshape(-1, a.shape[1]))  # experts x layers
```

See `MANIFEST.csv` for shape / top_k / observed expert count per model.

## Validity
- **16 valid** traces (listed OK in MANIFEST).
- **`traces/_INVALID/`** — `openai_gpt-oss-20b` and `gpt-oss-120b` captured **degenerate**
  (`observed_num_experts == 1`; the MoE config collapsed). **Do not use**; need re-capture.
- `meta-llama_Llama-4-Scout` has `top_k=1` (1 routed expert/token + an always-on shared
  expert that is not in this array) — valid but not a standard top-k trace.

## Pulling on another machine
```bash
git fetch origin expert-traces-data
git checkout expert-traces-data     # data-only orphan branch, no LLMServingSim code
```
This is an **orphan branch** of the `LLMServingSim` fork: it carries only these traces,
independent of the code branches.
