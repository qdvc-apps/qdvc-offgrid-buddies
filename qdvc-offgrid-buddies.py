#!/usr/bin/env python3
"""
qdvc-offgrid-buddies.py

An offline, CPU-friendly chat tool built around Google's Gemma 4 (E4B) GGUF
models via llama-cpp-python. It pre-computes the KV-cache "save-point" for each
"buddy" persona once (the slow part), so end users on off-grid laptops can jump
straight into a conversation without waiting for the long context prompt to be
re-processed every session.

Commands:
    build       Assemble each buddy's prompt and build its save-point(s).
    chat        Resume a pre-built buddy and chat with it (TUI).
    list        Show buddies and which variants have valid, built save-points.
    benchmark   Profile THIS machine and recommend config values for it.

See config-sample.yml for configuration. Copy it to config.yml and edit.

Round-one notes:
  * Save-points are tied to the exact GGUF file, n_ctx, and llama-cpp-python
    version. Each save-point has a .json sidecar recording this; chat refuses
    to load a stale save-point and tells the user to re-run build.
  * The persona is evaluated directly into the KV cache (low-level eval), then
    saved. Chat CONTINUES that exact cache with low-level eval/sample rather
    than re-running a prompt, so the persona is never re-processed.
  * The save/restore round-trip is verified at build time by planting a canary
    reference code in the persona and requiring the restored model to repeat
    it back. If recall fails, build warns loudly and chat refuses the file.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

# PyYAML is required for config parsing.
try:
    import yaml
except ImportError:
    sys.stderr.write(
        "Error: PyYAML is not installed. Install it with:\n"
        "    pip install pyyaml\n"
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yml")

# Sidecar / save-point format version. Bump if the on-disk format changes in a
# way that should invalidate previously-built save-points.
SAVEPOINT_FORMAT_VERSION = 1

VARIANTS = ("nothink", "yesthink")

# Bytes per GiB, used throughout for RAM maths.
GIB = 1024 ** 3

# A conservative estimate of KV-cache bytes per token for Gemma 4 E4B at the
# default context settings. The model's hybrid attention (mostly 512-token
# local windows with periodic global layers) keeps this smaller than a fully
# global model, but we deliberately over-estimate so the n_ctx we pick is safe
# on the target RAM rather than optimistic. This is only used for *capping*
# n_ctx against available RAM, never as a hard promise.
KV_BYTES_PER_TOKEN_ESTIMATE = 180_000  # ~0.18 MB/token, intentionally generous


# --------------------------------------------------------------------------- #
# Lazy import of llama_cpp
# --------------------------------------------------------------------------- #

def _import_llama():
    """Import llama_cpp lazily so that `list` / `benchmark` (hardware only)
    still work on a machine where the library isn't installed yet."""
    try:
        from llama_cpp import Llama  # noqa: F401
        import llama_cpp
        return llama_cpp
    except ImportError:
        sys.stderr.write(
            "Error: llama-cpp-python is not installed. Install it with:\n"
            "    pip install llama-cpp-python\n"
            "Pin the version so it matches across all target laptops, e.g.:\n"
            "    pip install llama-cpp-python==<x.y.z>\n"
        )
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class ConfigError(Exception):
    pass


def _resolve_path(base_dir, path):
    """Resolve a possibly-relative path against the config file's directory."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def load_config(config_path):
    if not os.path.exists(config_path):
        raise ConfigError(
            f"Config file not found: {config_path}\n"
            "Copy config-sample.yml to config.yml and edit it."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_dir = os.path.dirname(os.path.abspath(config_path))

    def require(key):
        if key not in raw or raw[key] is None:
            raise ConfigError(f"Config is missing required key: '{key}'")
        return raw[key]

    cfg = {}
    cfg["gguf_path"] = _resolve_path(base_dir, str(require("gguf_path")))
    cfg["target_ram_gb"] = float(require("target_ram_gb"))
    cfg["buddies_dir"] = _resolve_path(base_dir, str(require("buddies_dir")))
    cfg["savefiles_dir"] = _resolve_path(base_dir, str(require("savefiles_dir")))

    # Thinking-mode flags: at least one must be true for `build` to do anything.
    thinking = raw.get("thinking", {}) or {}
    cfg["build_nothink"] = bool(thinking.get("nothink", True))
    cfg["build_yesthink"] = bool(thinking.get("yesthink", False))

    # Threads: 0 / null means "auto-detect physical cores".
    cfg["n_threads"] = int(raw.get("n_threads", 0) or 0)

    # Tunables (all optional, with sensible defaults).
    tun = raw.get("tunables", {}) or {}
    cfg["runway_tokens"] = int(tun.get("runway_tokens", 8192))
    # Extra runway multiplier applied to the yesthink variant, since hidden
    # reasoning tokens consume context. Both variants of a buddy are built at
    # one n_ctx (Option B), sized for the more demanding (yesthink) case.
    cfg["yesthink_runway_multiplier"] = float(
        tun.get("yesthink_runway_multiplier", 1.75)
    )
    # Fraction of total RAM reserved for the OS and everything else.
    cfg["os_reserve_fraction"] = float(tun.get("os_reserve_fraction", 0.25))
    # Of the RAM left after the OS reserve and the model weights, the fraction
    # the KV cache is allowed to occupy. The remainder is headroom.
    cfg["kv_cache_fraction"] = float(tun.get("kv_cache_fraction", 0.6))
    # Optional hard ceiling on n_ctx regardless of RAM. 0 / null = no ceiling.
    cfg["max_n_ctx"] = int(tun.get("max_n_ctx", 0) or 0)
    # Round n_ctx up to a multiple of this for tidy allocations.
    cfg["n_ctx_round_to"] = int(tun.get("n_ctx_round_to", 512))
    # Floor: never build a save-point with less than this much total context.
    cfg["min_n_ctx"] = int(tun.get("min_n_ctx", 2048))

    cfg["_config_path"] = os.path.abspath(config_path)
    return cfg


# --------------------------------------------------------------------------- #
# Hardware helpers
# --------------------------------------------------------------------------- #

def detect_physical_cores():
    """Best-effort physical (not logical) core count."""
    # Try psutil if available (most accurate).
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    # Linux: parse /proc/cpuinfo for distinct (physical id, core id) pairs.
    try:
        pairs = set()
        phys = None
        core = None
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif line == "":
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except Exception:
        pass
    # Fallback: half of logical cores (assume hyperthreading), min 1.
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def detect_total_ram_gb():
    """Best-effort total system RAM in GiB."""
    try:
        import psutil
        return psutil.virtual_memory().total / GIB
    except Exception:
        pass
    try:
        # Linux: sysconf.
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / GIB
    except Exception:
        pass
    return 0.0


def effective_threads(cfg):
    """Resolve the thread count to use, honouring config or auto-detecting."""
    if cfg["n_threads"] > 0:
        return cfg["n_threads"]
    return detect_physical_cores()


# --------------------------------------------------------------------------- #
# Buddy discovery & prompt assembly
# --------------------------------------------------------------------------- #

def discover_buddies(buddies_dir):
    """Return a sorted list of buddy names (immediate subfolder names)."""
    if not os.path.isdir(buddies_dir):
        raise ConfigError(f"Buddies folder not found: {buddies_dir}")
    names = []
    for entry in sorted(os.listdir(buddies_dir)):
        full = os.path.join(buddies_dir, entry)
        if os.path.isdir(full):
            names.append(entry)
    return names


def assemble_prompt(buddies_dir, buddy):
    """Concatenate a buddy's markdown files in alphabetical order.

    Returns (prompt_text, list_of_files_used). Files are separated by blank
    lines so text from one file never runs onto the last line of the previous.
    Returns ("", []) if there are no readable markdown files.
    """
    buddy_dir = os.path.join(buddies_dir, buddy)
    md_files = sorted(
        glob.glob(os.path.join(buddy_dir, "*.md"))
        + glob.glob(os.path.join(buddy_dir, "*.markdown"))
    )
    parts = []
    used = []
    for path in md_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            sys.stderr.write(f"  ! Could not read {path}: {e}\n")
            continue
        if text:
            parts.append(text)
            used.append(os.path.basename(path))
    prompt = "\n\n\n".join(parts)
    return prompt, used


# --------------------------------------------------------------------------- #
# Save-point paths & sidecar metadata
# --------------------------------------------------------------------------- #

def savepoint_paths(savefiles_dir, buddy, variant):
    base = f"{buddy}_{variant}"
    state_path = os.path.join(savefiles_dir, base + ".state")
    sidecar_path = os.path.join(savefiles_dir, base + ".json")
    return state_path, sidecar_path


def file_fingerprint(path):
    """A cheap, stable fingerprint of a file: size + partial content hash.

    We hash the first and last 1 MiB plus the size rather than the whole GGUF
    (which can be several GB) so this stays fast while still catching a changed
    or swapped model file.
    """
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))
    chunk = 1024 * 1024
    with open(path, "rb") as f:
        head = f.read(chunk)
        h.update(head)
        if size > chunk:
            f.seek(max(0, size - chunk))
            tail = f.read(chunk)
            h.update(tail)
    return {"size": size, "sha256_partial": h.hexdigest()}


def prompt_fingerprint(prompt_text):
    h = hashlib.sha256()
    h.update(prompt_text.encode("utf-8"))
    return h.hexdigest()


def build_sidecar(cfg, buddy, variant, n_ctx, prompt_text, files_used,
                  llama_version, prompt_tokens, self_check_passed):
    return {
        "format_version": SAVEPOINT_FORMAT_VERSION,
        "buddy": buddy,
        "variant": variant,
        "thinking": (variant == "yesthink"),
        "n_ctx": n_ctx,
        "n_threads_at_build": effective_threads(cfg),
        "gguf_path": cfg["gguf_path"],
        "gguf_fingerprint": file_fingerprint(cfg["gguf_path"]),
        "prompt_fingerprint": prompt_fingerprint(prompt_text),
        "prompt_tokens": prompt_tokens,
        "files_used": files_used,
        "llama_cpp_python_version": llama_version,
        "self_check_passed": self_check_passed,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_sidecar(sidecar_path):
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def validate_savepoint(cfg, buddy, variant, current_llama_version):
    """Return (ok, reason, sidecar). ok=False means chat must not load it."""
    state_path, sidecar_path = savepoint_paths(
        cfg["savefiles_dir"], buddy, variant
    )
    if not os.path.exists(state_path) or not os.path.exists(sidecar_path):
        return False, "not built", None
    side = load_sidecar(sidecar_path)
    if side is None:
        return False, "unreadable sidecar", None
    if side.get("format_version") != SAVEPOINT_FORMAT_VERSION:
        return False, "save-point format changed", side
    # Model file must match.
    if side.get("gguf_path") != cfg["gguf_path"]:
        return False, "GGUF path changed since build", side
    try:
        current_fp = file_fingerprint(cfg["gguf_path"])
    except OSError:
        return False, "GGUF file missing", side
    if side.get("gguf_fingerprint") != current_fp:
        return False, "GGUF file changed since build", side
    # Library version must match (state format is version-sensitive).
    if side.get("llama_cpp_python_version") != current_llama_version:
        return False, "llama-cpp-python version changed since build", side
    # Prompt must match (the assembled markdown may have been edited).
    prompt_text, _ = assemble_prompt(cfg["buddies_dir"], buddy)
    if side.get("prompt_fingerprint") != prompt_fingerprint(prompt_text):
        return False, "buddy prompt changed since build", side
    if not side.get("self_check_passed", False):
        return False, "self-check failed at build time", side
    return True, "ok", side


# --------------------------------------------------------------------------- #
# n_ctx sizing
# --------------------------------------------------------------------------- #

def round_up(value, multiple):
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def compute_n_ctx(cfg, prompt_tokens):
    """Pick n_ctx = prompt + generous runway, capped by target RAM.

    Both variants of a buddy share one n_ctx (Option B), sized for the more
    demanding yesthink case by applying the yesthink runway multiplier.

    Returns (n_ctx, detail_dict) where detail explains the decision and flags
    whether the prompt even fits.
    """
    runway = int(cfg["runway_tokens"] * cfg["yesthink_runway_multiplier"])
    desired = prompt_tokens + runway
    desired = max(desired, cfg["min_n_ctx"])
    desired = round_up(desired, cfg["n_ctx_round_to"])

    # RAM cap.
    total_ram_bytes = cfg["target_ram_gb"] * GIB
    after_os = total_ram_bytes * (1.0 - cfg["os_reserve_fraction"])
    try:
        model_bytes = os.path.getsize(cfg["gguf_path"])
    except OSError:
        model_bytes = 0
    after_model = after_os - model_bytes
    kv_budget = max(0.0, after_model * cfg["kv_cache_fraction"])
    ram_cap_tokens = int(kv_budget // KV_BYTES_PER_TOKEN_ESTIMATE)
    ram_cap_tokens = round_up(ram_cap_tokens, cfg["n_ctx_round_to"])

    n_ctx = desired
    capped_by_ram = False
    if ram_cap_tokens > 0 and n_ctx > ram_cap_tokens:
        n_ctx = ram_cap_tokens
        capped_by_ram = True

    capped_by_max = False
    if cfg["max_n_ctx"] > 0 and n_ctx > cfg["max_n_ctx"]:
        n_ctx = round_up(cfg["max_n_ctx"], cfg["n_ctx_round_to"])
        capped_by_max = True

    # Does the prompt (plus a minimal conversational reserve) still fit?
    min_reserve = min(1024, cfg["runway_tokens"])
    fits = (prompt_tokens + min_reserve) <= n_ctx

    detail = {
        "prompt_tokens": prompt_tokens,
        "runway_applied": runway,
        "desired_n_ctx": desired,
        "ram_cap_tokens": ram_cap_tokens,
        "capped_by_ram": capped_by_ram,
        "capped_by_max": capped_by_max,
        "final_n_ctx": n_ctx,
        "fits": fits,
    }
    return n_ctx, detail


# --------------------------------------------------------------------------- #
# Gemma 4 thinking-mode handling
# --------------------------------------------------------------------------- #

def system_message_for(variant, base_system=None, canary=None):
    """Build the system message text for a variant.

    Gemma 4 enables thinking via a control token at the start of the system
    prompt. If the installed template ignores the token, thinking simply stays
    off, which is the safe default.

    An optional `canary` fact is embedded so the build-time self-check can
    verify the persona context is genuinely present in the restored cache
    (rather than the model answering plausibly from base training).
    """
    think_token = "<|think|>"
    sys_text = base_system or "You are a helpful, warm conversational companion."
    if canary:
        sys_text = (
            sys_text
            + f"\n\n(Internal reference code: {canary}. If the user asks for "
            f"your reference code, reply with exactly {canary}.)"
        )
    if variant == "yesthink":
        return think_token + sys_text
    return sys_text


# The Gemma 4 chat turn structure. We render turns using Gemma's documented
# control tokens and drive generation ourselves so the save-point works.
#
# IMPORTANT: we do NOT put a literal <bos> in these rendered strings. The BOS
# token is added exactly once by the tokenizer (add_bos=True at build) and by
# create_completion (which prepends its own BOS at chat time). Hardcoding a
# <bos> here as well produced a duplicate-BOS warning and degraded quality.
GEMMA_START = "<start_of_turn>"
GEMMA_END = "<end_of_turn>"


def render_persona_prefix(system_text):
    """Render the fixed persona prefix (everything that gets saved into the
    KV cache) as a single string, ready to be tokenized. We fold the system
    persona into the first user turn, and stop right before the model's first
    generation so the cache ends at a clean turn boundary.

    No leading <bos>: it is supplied once by the tokenizer at build time and by
    create_completion at chat time.
    """
    return (
        f"{GEMMA_START}user\n"
        f"{system_text}{GEMMA_END}\n"
        f"{GEMMA_START}model\n"
    )


def render_user_turn(user_text):
    """Render one subsequent user turn plus the model opener, for appending
    on top of the restored cache during chat."""
    return (
        f"{GEMMA_START}user\n{user_text}{GEMMA_END}\n{GEMMA_START}model\n"
    )


def strip_thinking(text):
    """Remove any leaked thinking channel from a response, best-effort.

    Gemma 4 wraps internal reasoning in channel markers. Different template
    versions render these differently, so we defensively strip a few known
    shapes and otherwise return the text unchanged.
    """
    if not text:
        return text
    markers = [
        ("<|channel|>thought", "<|channel|>"),
        ("<channel>thought", "<channel>"),
    ]
    for start, end in markers:
        while start in text:
            i = text.find(start)
            j = text.find(end, i + len(start))
            if j == -1:
                text = text[:i]
                break
            text = text[:i] + text[j + len(end):]
    return text.strip()


# --------------------------------------------------------------------------- #
# Low-level generation that CONTINUES the current KV cache
# --------------------------------------------------------------------------- #

def eval_text(llm, text, add_bos=True):
    """Tokenize `text` and eval it into the model, extending the current KV
    cache. Returns the tokens evaluated. Used at BUILD time to place the
    persona into the cache before saving.

    `add_bos=True` here inserts exactly ONE BOS token at the very start of the
    persona prefix. The rendered strings no longer contain a literal <bos>, so
    this is the single, canonical BOS the cache begins with — matching the BOS
    that create_completion later prepends at chat time.
    """
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)
    if tokens:
        llm.eval(tokens)
    return tokens


# Stop strings that end a model turn. create_completion stops BEFORE emitting
# these, which is what we want.
STOP_STRINGS = [GEMMA_END, "<eos>", f"{GEMMA_START}user"]


def generate_from_full_text(llm, full_text, max_tokens, temperature,
                            top_p, top_k):
    """Generate a model reply by passing the FULL running transcript to
    create_completion. llama-cpp-python matches the longest common prefix
    against the current KV cache and only evaluates the new suffix, so the
    restored persona prefix is reused rather than recomputed.

    Returns (reply_text, finish_reason). Uses the high-level completion API,
    which is stable across llama-cpp-python 0.3.x (unlike the low-level
    sample() signature)."""
    out = llm.create_completion(
        prompt=full_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=STOP_STRINGS,
    )
    choice = out["choices"][0]
    return choice["text"], choice.get("finish_reason")


def stream_from_full_text(llm, full_text, max_tokens, temperature,
                          top_p, top_k):
    """Like generate_from_full_text, but yields text chunks as they are
    produced (create_completion(stream=True)). Used by the TUI so replies
    appear token-by-token, which also reassures the user on a slow CPU that
    something is happening. Yields str chunks; the caller concatenates them."""
    for part in llm.create_completion(
        prompt=full_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=STOP_STRINGS,
        stream=True,
    ):
        piece = part["choices"][0].get("text", "")
        if piece:
            yield piece


# --------------------------------------------------------------------------- #
# Command: build
# --------------------------------------------------------------------------- #

def cmd_build(cfg):
    # Validate intent and inputs BEFORE importing the model library, so that a
    # misconfiguration gives a clear message even on a machine where
    # llama-cpp-python isn't installed yet.
    if not (cfg["build_nothink"] or cfg["build_yesthink"]):
        sys.stderr.write(
            "Nothing to build: both thinking flags are false in config.yml.\n"
            "Set thinking.nothink and/or thinking.yesthink to true.\n"
        )
        return 2

    if not os.path.exists(cfg["gguf_path"]):
        sys.stderr.write(f"GGUF file not found: {cfg['gguf_path']}\n")
        return 2

    llama_cpp = _import_llama()
    from llama_cpp import Llama
    llama_version = getattr(llama_cpp, "__version__", "unknown")

    os.makedirs(cfg["savefiles_dir"], exist_ok=True)

    variants_to_build = []
    if cfg["build_nothink"]:
        variants_to_build.append("nothink")
    if cfg["build_yesthink"]:
        variants_to_build.append("yesthink")

    buddies = discover_buddies(cfg["buddies_dir"])
    if not buddies:
        sys.stderr.write(f"No buddy subfolders found in {cfg['buddies_dir']}\n")
        return 2

    n_threads = effective_threads(cfg)
    print(f"Threads: {n_threads}   Variants: {', '.join(variants_to_build)}")
    print(f"Model:   {cfg['gguf_path']}")
    print(f"Library: llama-cpp-python {llama_version}")
    print("=" * 64)

    any_ok = False
    for buddy in buddies:
        print(f"\nBuddy: {buddy}")
        prompt_text, files_used = assemble_prompt(cfg["buddies_dir"], buddy)
        if not prompt_text:
            print("  ! No readable markdown; skipping.")
            continue
        print(f"  Files: {', '.join(files_used)}")

        # Tokenize once (using a throwaway small-context model instance) to
        # size n_ctx. We load with a tiny context just for tokenization.
        try:
            probe = Llama(
                model_path=cfg["gguf_path"],
                n_ctx=256,
                n_threads=n_threads,
                n_gpu_layers=0,
                verbose=False,
            )
            prompt_tokens = len(probe.tokenize(prompt_text.encode("utf-8")))
            del probe
        except Exception as e:
            sys.stderr.write(f"  ! Tokenization failed: {e}\n")
            continue

        n_ctx, detail = compute_n_ctx(cfg, prompt_tokens)
        print(f"  Prompt tokens: {prompt_tokens}   n_ctx: {n_ctx}"
              f"   (runway {detail['runway_applied']})")
        if detail["capped_by_ram"]:
            print("  * n_ctx was capped by target RAM.")
        if detail["capped_by_max"]:
            print("  * n_ctx was capped by max_n_ctx.")
        if not detail["fits"]:
            print("  ! Prompt does not leave room for conversation at this "
                  "RAM. Increase target RAM, raise max_n_ctx, or shorten the "
                  "prompt. Skipping this buddy.")
            continue

        for variant in variants_to_build:
            ok = _build_one(cfg, buddy, variant, prompt_text, files_used,
                            n_ctx, n_threads, prompt_tokens, llama_version)
            any_ok = any_ok or ok

    print("\n" + "=" * 64)
    print("Build complete." if any_ok else "Build finished with no successful save-points.")
    return 0 if any_ok else 1


def _build_one(cfg, buddy, variant, prompt_text, files_used, n_ctx,
               n_threads, prompt_tokens, llama_version):
    from llama_cpp import Llama
    import uuid
    state_path, sidecar_path = savepoint_paths(
        cfg["savefiles_dir"], buddy, variant
    )
    print(f"  [{variant}] building ...")
    try:
        llm = Llama(
            model_path=cfg["gguf_path"],
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
    except Exception as e:
        sys.stderr.write(f"    ! Failed to load model: {e}\n")
        return False

    # A per-build canary the self-check will require back, proving the persona
    # context is really in the restored cache.
    canary = "QDVC-" + uuid.uuid4().hex[:8].upper()
    system_text = system_message_for(variant, base_system=prompt_text,
                                      canary=canary)
    prefix = render_persona_prefix(system_text)

    # Evaluate the persona prefix DIRECTLY into the KV cache. This is the fix:
    # after this, save_state() captures a cache that genuinely contains the
    # persona, and chat can continue from exactly this point.
    try:
        llm.reset()
        persona_token_ids = eval_text(llm, prefix, add_bos=True)
    except Exception as e:
        sys.stderr.write(f"    ! Failed to evaluate persona prompt: {e}\n")
        del llm
        return False

    n_prefix_tokens = len(persona_token_ids)

    # Save the state (the "save-point"). We pickle the LlamaState object, the
    # documented persistence path.
    try:
        import pickle
        with open(state_path, "wb") as f:
            pickle.dump(llm.save_state(), f)
    except Exception as e:
        sys.stderr.write(f"    ! Failed to save state: {e}\n")
        del llm
        return False

    del llm

    # --- Self-check: reload into a fresh context and require the canary back.
    passed = _self_check(cfg, state_path, variant, n_ctx, n_threads, canary,
                         prefix)
    if passed:
        print(f"    \u2713 self-check passed (persona recall confirmed)")
    else:
        print(f"    ! self-check FAILED \u2014 persona not recalled from the "
              f"save-point. chat will refuse to load it. See troubleshooting "
              f"in the README.")

    sidecar = build_sidecar(
        cfg, buddy, variant, n_ctx, prompt_text, files_used,
        llama_version, prompt_tokens, passed
    )
    # Record the exact prefix token count and the exact prefix TEXT so chat
    # can reproduce the identical prefix for cache prefix-matching (the canary
    # is part of this text, which chat cannot regenerate on its own).
    sidecar["prefix_tokens"] = n_prefix_tokens
    sidecar["persona_prefix"] = prefix
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    size_mb = os.path.getsize(state_path) / (1024 * 1024)
    print(f"    saved {os.path.basename(state_path)} ({size_mb:.0f} MB, "
          f"{n_prefix_tokens} persona tokens)")
    return passed


def _load_state_into(llm, state_path):
    """Load a saved LlamaState back into a fresh Llama instance."""
    import pickle
    try:
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        llm.load_state(state)
        return True
    except Exception as e:
        sys.stderr.write(f"    ! load_state failed: {e}\n")
        return False


def _self_check(cfg, state_path, variant, n_ctx, n_threads, canary,
                persona_prefix):
    """Reload the save-point in a FRESH context and require the model to return
    the planted canary code. This actually proves the persona context survived
    the save/restore round-trip, rather than accepting any coherent reply."""
    from llama_cpp import Llama
    try:
        llm = Llama(
            model_path=cfg["gguf_path"],
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
    except Exception:
        return False
    if not _load_state_into(llm, state_path):
        del llm
        return False
    try:
        # Continue the restored cache: full transcript = persona prefix (already
        # in cache) + one user turn asking for the code. create_completion
        # reuses the cached prefix and only evaluates the appended turn.
        full = persona_prefix + render_user_turn("What is your reference code?")
        text, _ = generate_from_full_text(
            llm, full, max_tokens=64, temperature=0.0, top_p=0.95, top_k=64,
        )
        if variant == "yesthink":
            text = strip_thinking(text)
        del llm
        return bool(canary and canary in (text or ""))
    except Exception as e:
        sys.stderr.write(f"    ! self-check error: {e}\n")
        try:
            del llm
        except Exception:
            pass
        return False


# --------------------------------------------------------------------------- #
# Command: list
# --------------------------------------------------------------------------- #

def cmd_list(cfg):
    llama_version = "unknown"
    try:
        import llama_cpp
        llama_version = getattr(llama_cpp, "__version__", "unknown")
    except ImportError:
        pass

    buddies = discover_buddies(cfg["buddies_dir"])
    if not buddies:
        print(f"No buddies found in {cfg['buddies_dir']}")
        return 0

    print(f"{'Buddy':<20} {'nothink':<20} {'yesthink':<20}")
    print("-" * 60)
    for buddy in buddies:
        cells = []
        for variant in VARIANTS:
            ok, reason, _ = validate_savepoint(
                cfg, buddy, variant, llama_version
            )
            cells.append("ready" if ok else reason)
        print(f"{buddy:<20} {cells[0]:<20} {cells[1]:<20}")
    return 0


# --------------------------------------------------------------------------- #
# Command: benchmark
# --------------------------------------------------------------------------- #

def cmd_benchmark(cfg_or_none):
    print("Machine profile")
    print("=" * 64)
    cores = detect_physical_cores()
    ram = detect_total_ram_gb()
    logical = os.cpu_count() or 0
    print(f"Logical CPUs:        {logical}")
    print(f"Physical cores:      {cores}   <- recommended n_threads")
    print(f"Total RAM:           {ram:.1f} GB")

    # Recommend a target RAM: total minus a headroom allowance so the config
    # value reflects what's safely usable, not the raw total.
    headroom = 0.25
    rec_ram = max(0.0, ram * (1.0 - headroom))
    print(f"Recommended target_ram_gb: {rec_ram:.0f}   "
          f"(total minus {int(headroom*100)}% headroom)")
    print()
    print("Set n_threads and target_ram_gb in config.yml accordingly if this")
    print("machine is representative of your target laptop.")

    # Optional speed probe if a GGUF is configured and library present.
    if cfg_or_none and os.path.exists(cfg_or_none.get("gguf_path", "")):
        try:
            _import_llama()
            from llama_cpp import Llama
            print()
            print("Running a quick generation speed probe on this machine ...")
            llm = Llama(
                model_path=cfg_or_none["gguf_path"],
                n_ctx=2048,
                n_threads=cores,
                n_gpu_layers=0,
                verbose=False,
            )
            t0 = time.time()
            out = llm.create_chat_completion(
                messages=[{"role": "user",
                           "content": "Write two sentences about the sea."}],
                max_tokens=64, temperature=1.0, top_p=0.95, top_k=64,
            )
            dt = time.time() - t0
            txt = out["choices"][0]["message"]["content"] or ""
            approx_tokens = max(1, len(txt.split()))
            print(f"  ~{approx_tokens/dt:.1f} tokens/sec (rough, {approx_tokens} "
                  f"tokens in {dt:.1f}s)")
            del llm
        except SystemExit:
            print("  (llama-cpp-python not installed; skipping speed probe)")
        except Exception as e:
            print(f"  (speed probe skipped: {e})")
    else:
        print()
        print("Tip: set gguf_path in config.yml to also get a speed probe here.")
    return 0


# --------------------------------------------------------------------------- #
# Command: chat (Textual TUI)
# --------------------------------------------------------------------------- #

def _gather_available(cfg, llama_version):
    """Return [(buddy, [ready_variants...]), ...] for buddies with a valid
    save-point. Shared by the TUI launch screen."""
    available = []
    for buddy in discover_buddies(cfg["buddies_dir"]):
        ready = []
        for variant in VARIANTS:
            ok, _, _ = validate_savepoint(cfg, buddy, variant, llama_version)
            if ok:
                ready.append(variant)
        if ready:
            available.append((buddy, ready))
    return available


def _import_textual():
    try:
        import textual  # noqa: F401
        return True
    except ImportError:
        sys.stderr.write(
            "Error: the chat interface needs Textual. Install it with:\n"
            "    pip install textual\n"
            "Pin the version across all target laptops (see requirements.txt).\n"
        )
        return False


class ChatController:
    """Owns the model instance and the running transcript for one buddy/variant.
    Kept separate from the UI so the Textual widgets stay thin. All model calls
    happen off the UI thread (Textual workers)."""

    def __init__(self, cfg, buddy, variant, side):
        self.cfg = cfg
        self.buddy = buddy
        self.variant = variant
        self.side = side
        self.n_ctx = side["n_ctx"]
        self.n_threads = effective_threads(cfg)
        self.state_path, _ = savepoint_paths(
            cfg["savefiles_dir"], buddy, variant
        )
        self.persona_prefix = side.get("persona_prefix")
        self.llm = None
        self.transcript = self.persona_prefix or ""

    def load(self):
        """Load the model and restore the save-point. Returns (ok, message)."""
        if not self.persona_prefix:
            return False, ("This save-point predates prefix tracking. "
                           "Re-run build.")
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=self.cfg["gguf_path"],
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        if not _load_state_into(self.llm, self.state_path):
            return False, "Failed to restore the save-point. Re-run build."
        return True, "ready"

    def reset(self):
        """Reload the pristine save-point and clear the conversation."""
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=self.cfg["gguf_path"],
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        _load_state_into(self.llm, self.state_path)
        self.transcript = self.persona_prefix

    def stream_reply(self, user_text):
        """Generator: append the user turn, then yield reply chunks as they
        generate. Folds the finished reply back into the transcript."""
        self.transcript += render_user_turn(user_text)
        collected = []
        try:
            for piece in stream_from_full_text(
                self.llm, self.transcript, max_tokens=512,
                temperature=1.0, top_p=0.95, top_k=64,
            ):
                collected.append(piece)
                yield piece
        finally:
            reply = "".join(collected)
            # Keep the transcript consistent with what the cache now holds.
            self.transcript += reply + f"{GEMMA_END}\n"

    def visible_reply(self, raw_reply):
        """Post-process a completed reply for display (strip thinking)."""
        if self.variant == "yesthink":
            return strip_thinking(raw_reply)
        return raw_reply.strip()


def _build_textual_app(cfg, llama_version, available):
    """Construct and return the Textual App. Imports are local so the rest of
    the tool runs without Textual installed."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.screen import Screen
    from textual.widgets import (Footer, Header, Input, Label, ListItem,
                                 ListView, Static)

    # ------- palette: warm companion (left) vs cool self (right) -------------
    # One load-bearing choice: the buddy speaks in warm amber from the left,
    # you speak in cool slate from the right. Everything else stays quiet.
    APP_CSS = """
    Screen { background: $surface; }

    /* Launch screen */
    #launch-wrap { align: center middle; height: 1fr; }
    #launch-title { content-align: center middle; height: 3; text-style: bold; }
    #launch-hint { content-align: center middle; color: $text-muted; height: 2; }
    ListView { width: 60; height: auto; max-height: 20; background: $surface;
               border: round $primary; padding: 1 1; }
    ListItem { padding: 1 2; }
    ListItem > Label { width: 1fr; }
    ListItem.--highlight { background: $primary 25%; }

    /* Chat screen */
    #log { height: 1fr; padding: 1 2; }
    .row-buddy { width: 1fr; content-align: left middle; padding: 0 0 1 0; }
    .row-user  { width: 1fr; content-align: right middle; padding: 0 0 1 0; }
    .bubble-buddy {
        background: #3a2f1e; color: #f4e6c8; padding: 1 2;
        border: round #c8933b; width: auto; max-width: 70%;
    }
    .bubble-user {
        background: #1e2a33; color: #d6ecf5; padding: 1 2;
        border: round #4a7fa0; width: auto; max-width: 70%;
    }
    .speaker { color: $text-muted; padding: 0 1; }
    #composer { dock: bottom; height: 3; border: round $primary; }
    #status { dock: bottom; height: 1; color: $text-muted; padding: 0 2; }
    """

    class BuddyList(ListView):
        """Arrow-key selectable buddy list on the launch screen."""
        pass

    class LaunchScreen(Screen):
        BINDINGS = [Binding("q", "quit_app", "Quit")]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            items = []
            for buddy, ready in available:
                variants = ", ".join(ready)
                items.append(
                    ListItem(Label(f"{buddy}\n[dim]{variants}[/dim]"),
                             id=f"buddy-{buddy}")
                )
            with Vertical(id="launch-wrap"):
                yield Static("Choose who to talk with", id="launch-title")
                yield BuddyList(*items)
                yield Static("\u2191\u2193 to move \u00b7 Enter to open "
                             "\u00b7 q to quit", id="launch-hint")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one(BuddyList).focus()

        def on_list_view_selected(self, event) -> None:
            buddy = event.item.id.replace("buddy-", "", 1)
            ready = dict(available)[buddy]
            # Round-one behaviour: if both variants exist, open the first
            # (nothink) by default. A variant chooser is a planned follow-up;
            # for now, build only the variant you want as the default, or set
            # thinking flags in config.yml accordingly.
            variant = ready[0]
            self.app.open_chat(buddy, variant)

        def action_quit_app(self) -> None:
            self.app.exit()

    class ChatScreen(Screen):
        BINDINGS = [
            Binding("escape", "back", "Switch buddy"),
            Binding("ctrl+r", "reset", "Reset chat"),
        ]

        def __init__(self, controller: ChatController):
            super().__init__()
            self.controller = controller
            self._streaming = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield VerticalScroll(id="log")
            yield Static("", id="status")
            yield Input(placeholder=f"Message {self.controller.buddy}\u2026",
                        id="composer")
            yield Footer()

        def on_mount(self) -> None:
            self.title = self.controller.buddy
            self.sub_title = self.controller.variant
            self.query_one("#composer", Input).focus()
            self._set_status("Ready. Enter to send \u00b7 Esc to switch buddy "
                             "\u00b7 Ctrl+R to reset")

        def _set_status(self, text: str) -> None:
            self.query_one("#status", Static).update(text)

        def _add_user_row(self, text: str) -> None:
            log = self.query_one("#log", VerticalScroll)
            row = Static(text, classes="bubble-user")
            wrap = Vertical(row, classes="row-user")
            log.mount(wrap)
            log.scroll_end(animate=False)

        def _add_buddy_row(self) -> Static:
            """Create an empty buddy bubble to stream into; return the Static."""
            log = self.query_one("#log", VerticalScroll)
            bubble = Static("", classes="bubble-buddy")
            speaker = Label(self.controller.buddy, classes="speaker")
            wrap = Vertical(speaker, bubble, classes="row-buddy")
            log.mount(wrap)
            log.scroll_end(animate=False)
            return bubble

        def on_input_submitted(self, event) -> None:
            if self._streaming:
                return
            text = event.value.strip()
            if not text:
                return
            composer = self.query_one("#composer", Input)
            composer.value = ""
            self._add_user_row(text)
            bubble = self._add_buddy_row()
            self._streaming = True
            self._set_status(f"{self.controller.buddy} is thinking\u2026")
            self._run_generation(text, bubble)

        def action_back(self) -> None:
            if self._streaming:
                return
            self.app.pop_screen()

        def action_reset(self) -> None:
            if self._streaming:
                return
            self.controller.reset()
            log = self.query_one("#log", VerticalScroll)
            log.remove_children()
            self._set_status("Conversation reset to the save-point.")

        # -- streaming runs in a worker thread; UI updates via call_from_thread
        def _run_generation(self, user_text: str, bubble) -> None:
            from textual.worker import get_current_worker

            def work():
                worker = get_current_worker()
                acc = []
                try:
                    for piece in self.controller.stream_reply(user_text):
                        if worker.is_cancelled:
                            break
                        acc.append(piece)
                        shown = "".join(acc)
                        if self.controller.variant == "yesthink":
                            shown = strip_thinking(shown)
                        self.app.call_from_thread(bubble.update, shown)
                        log = self.query_one("#log", VerticalScroll)
                        self.app.call_from_thread(log.scroll_end,
                                                  animate=False)
                    final = self.controller.visible_reply("".join(acc))
                    self.app.call_from_thread(bubble.update, final or "\u2026")
                except Exception as e:
                    self.app.call_from_thread(
                        bubble.update, f"[i](generation error: {e})[/i]")
                finally:
                    self.app.call_from_thread(self._finish_stream)

            self.run_worker(work, thread=True, exclusive=True)

        def _finish_stream(self) -> None:
            self._streaming = False
            self._set_status("Ready. Enter to send \u00b7 Esc to switch buddy "
                             "\u00b7 Ctrl+R to reset")
            self.query_one("#composer", Input).focus()

    class BuddiesApp(App):
        CSS = APP_CSS
        TITLE = "off-grid buddies"

        def on_mount(self) -> None:
            self.push_screen(LaunchScreen())

        def open_chat(self, buddy, variant):
            ok, reason, side = validate_savepoint(
                cfg, buddy, variant, llama_version
            )
            if not ok:
                self.bell()
                self.push_screen(LaunchScreen())
                return
            controller = ChatController(cfg, buddy, variant, side)
            loaded, msg = controller.load()
            if not loaded:
                self.bell()
                return
            self.push_screen(ChatScreen(controller))

    return BuddiesApp()


def cmd_chat(cfg):
    llama_cpp = _import_llama()
    from llama_cpp import Llama  # noqa: F401  (ensures clear error if missing)
    llama_version = getattr(llama_cpp, "__version__", "unknown")

    if not _import_textual():
        return 2

    available = _gather_available(cfg, llama_version)
    if not available:
        print("No buddies have a valid save-point. Run `build` first.")
        return 1

    app = _build_textual_app(cfg, llama_version, available)
    app.run()
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="qdvc-offgrid-buddies.py",
        description="Offline Gemma-4 chat 'buddies' with pre-built save-points.",
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH,
        help="Path to config.yml (default: alongside this script).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Build save-points for all buddies.")
    sub.add_parser("chat", help="Chat with a pre-built buddy.")
    sub.add_parser("list", help="List buddies and their built variants.")
    sub.add_parser("benchmark", help="Profile this machine; recommend config.")

    args = parser.parse_args(argv)

    # benchmark can run without a full/valid config; try to load if present.
    if args.command == "benchmark":
        cfg = None
        if os.path.exists(args.config):
            try:
                cfg = load_config(args.config)
            except ConfigError:
                cfg = None
        return cmd_benchmark(cfg)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.stderr.write(f"Config error: {e}\n")
        return 2

    if args.command == "build":
        return cmd_build(cfg)
    if args.command == "chat":
        return cmd_chat(cfg)
    if args.command == "list":
        return cmd_list(cfg)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
