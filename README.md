# qdvc-offgrid-buddies

An offline, CPU-friendly chat tool for off-grid laptops. It runs Google's
Gemma 4 (E4B) locally via `llama-cpp-python` and pre-computes each persona's
context so end users can start chatting instantly — no waiting for a long
prompt to be re-processed every session.

The trick is a **save-point**: the model's KV-cache state, captured once after
the persona's context prompt is loaded, then restored on demand. Building is
slow (done once, at provisioning); chatting is fast.

## How it works

Each **buddy** is a subfolder of `buddies/` (e.g. `comedian/`, `comforter/`,
`philosopher/`). Its markdown files are read in alphabetical order and joined
into one context prompt. `build` loads that prompt into the model, warms it,
and writes a save-point plus a small `.json` sidecar recording exactly what it
was built against. `chat` restores a save-point and drops you into a
conversation.

Two thinking-mode variants can be built per buddy:

- **nothink** — faster and livelier; recommended for persona chat.
- **yesthink** — Gemma 4's step-by-step reasoning; slower, more deliberate.

## Setup

1. Install Python 3.9+ and the dependencies:

   ```
   pip install -r requirements.txt
   ```

   Pin `llama-cpp-python` to one version and use it on every laptop (see
   `requirements.txt` for why).

2. Download a Gemma 4 E4B GGUF (e.g. `UD-Q4_K_XL`) and note its path.

3. Copy the sample config and edit it:

   ```
   cp config-sample.yml config.yml
   ```

## Commands

- `python qdvc-offgrid-buddies.py benchmark`
  Profiles this machine and recommends `n_threads` and `target_ram_gb`. Does
  not modify your config — you copy the values in yourself.

- `python qdvc-offgrid-buddies.py build`
  Assembles every buddy's prompt and builds the requested save-point
  variant(s). Sizes `n_ctx` from the prompt length, a generous conversational
  runway, and the target RAM. Runs a self-check on each save-point.

- `python qdvc-offgrid-buddies.py list`
  Shows each buddy and whether its `nothink` / `yesthink` variants are ready
  (or why they aren't).

- `python qdvc-offgrid-buddies.py chat`
  Pick a buddy (and, if both were built, a thinking mode) and chat.
  In-chat commands: `/reset` restarts from the save-point; `/quit` exits.

## Sizing notes

`n_ctx` is chosen so that **prompt tokens + conversational runway** fit, capped
so the KV cache fits safely in the target RAM alongside the model. `n_ctx` is a
runtime cost paid on every load, so it is sized to real need rather than maxed.
Because both variants of a buddy share one `n_ctx` (sized for the more
demanding `yesthink` case), `nothink` save-points are marginally larger than
strictly necessary — a deliberate simplification for this first round.

## How the cache is built and continued

The persona prompt is evaluated **directly** into the KV cache at the token
level, then saved. Chat restores that cache and **continues** it with
low-level eval/sample — each user turn's tokens are appended and the reply is
sampled from the restored state. The persona is never re-processed, and it is
never routed through `create_chat_completion` (which manages its own context
and bypasses the restored cache — that was the cause of the "buddy has no
memory of its prompt" bug in the first build).

## Important caveat & troubleshooting

The round-trip is **verified at build time** by planting a canary reference
code inside each persona and requiring the restored model to repeat it back.
If it can't, `build` reports the self-check failed and `chat` refuses that
save-point. Always confirm `list` shows "ready" on a representative machine
before provisioning the fleet.

If the self-check fails:

- Confirm your `llama-cpp-python` version matches what you built with — the
  saved state format is version-sensitive. Pin one version fleet-wide.
- Confirm you are on CPU (`n_gpu_layers=0`, which this tool forces).
  Save/restore is known-good on CPU; the historical garbage-output bug was
  GPU-only.
- If it still fails, the remaining suspect is Gemma 4's hybrid/sliding-window
  attention interacting with state save/restore in your specific build.

## Save-point validity

A save-point is invalidated (and `chat` will refuse it, pointing you back to
`build`) if any of these change: the GGUF file, the assembled buddy prompt, the
`n_ctx`, or the `llama-cpp-python` version. The `.json` sidecar records all of
these, plus the exact persona prefix text and its token count, which `chat`
needs to continue the restored cache correctly.

## Verifying the fix from the earlier round

If you previously saw buddies with "no memory" of their prompt, that was caused
by loading the persona through `create_chat_completion` at build time, which
does not leave the prompt in the KV cache for `save_state` to capture. This is
fixed: the persona is now evaluated directly into the cache. To confirm on your
hardware, just run `build` and watch for `self-check passed (persona recall
confirmed)` on each buddy, then `list` should show `ready`. The self-check
plants a hidden reference code in the persona and requires the restored model
to repeat it back, so a pass genuinely proves the context survived the
save/restore round-trip.
