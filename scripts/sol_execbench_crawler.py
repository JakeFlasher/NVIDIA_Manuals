from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "https://research.nvidia.com/benchmarks/sol-execbench/kernel/{}"

CLIPBOARD_CAPTURE_INIT_SCRIPT = r"""
(() => {
  const state = { last: "", all: [] };

  const record = (text) => {
    if (typeof text !== "string" || !text) return;
    state.last = text;
    state.all.push(text);
  };

  Object.defineProperty(window, "__sol_last_copied_text", {
    configurable: true,
    get() { return state.last; },
    set(v) { state.last = typeof v === "string" ? v : ""; },
  });

  Object.defineProperty(window, "__sol_all_copied_texts", {
    configurable: true,
    get() { return state.all; },
  });

  try {
    const clip = navigator.clipboard;
    if (clip && clip.writeText && !clip.writeText.__solWrapped) {
      const origWriteText = clip.writeText.bind(clip);
      const wrappedWriteText = async (text) => {
        record(text);
        return origWriteText(text);
      };
      wrappedWriteText.__solWrapped = true;
      clip.writeText = wrappedWriteText;
    }

    if (clip && clip.write && !clip.write.__solWrapped) {
      const origWrite = clip.write.bind(clip);
      const wrappedWrite = async (items) => {
        try {
          for (const item of items || []) {
            if (item.types && item.types.includes("text/plain")) {
              const blob = await item.getType("text/plain");
              const text = await blob.text();
              record(text);
              break;
            }
          }
        } catch (_) {}
        return origWrite(items);
      };
      wrappedWrite.__solWrapped = true;
      clip.write = wrappedWrite;
    }
  } catch (_) {}

  try {
    if (typeof DataTransfer !== "undefined" &&
        DataTransfer.prototype &&
        DataTransfer.prototype.setData &&
        !DataTransfer.prototype.setData.__solWrapped) {
      const origSetData = DataTransfer.prototype.setData;
      const wrappedSetData = function(type, data) {
        try {
          const t = String(type || "").toLowerCase();
          if ((t === "text/plain" || t === "text") && typeof data === "string") {
            record(data);
          }
        } catch (_) {}
        return origSetData.apply(this, arguments);
      };
      wrappedSetData.__solWrapped = true;
      DataTransfer.prototype.setData = wrappedSetData;
    }
  } catch (_) {}

  try {
    if (document.execCommand && !document.execCommand.__solWrapped) {
      const origExecCommand = document.execCommand.bind(document);
      const wrappedExecCommand = function(command) {
        const before = state.last;
        const result = origExecCommand.apply(document, arguments);
        try {
          if (String(command || "").toLowerCase() === "copy" && state.last === before) {
            const sel = document.getSelection ? String(document.getSelection() || "") : "";
            if (sel) record(sel);
          }
        } catch (_) {}
        return result;
      };
      wrappedExecCommand.__solWrapped = true;
      document.execCommand = wrappedExecCommand;
    }
  } catch (_) {}
})();
"""

CLEAN_UNCOPIED_CODE_JS = r"""
() => {
  const pres = document.querySelectorAll('pre:not([data-sol-copied="1"])');
  let fixed = 0;

  for (const pre of pres) {
    const code = pre.querySelector('code') || pre;

    const rawAttrs = [
      'data-code', 'data-raw', 'data-clipboard-text', 'data-source',
      'data-content', 'data-original',
    ];
    let rawFound = false;

    for (const el of [pre, code, pre.parentElement]) {
      if (!el) continue;
      for (const attr of rawAttrs) {
        const val = el.getAttribute(attr);
        if (val && val.trim().length > 10) {
          pre.textContent = val;
          pre.setAttribute('data-sol-copied', '1');
          fixed++;
          rawFound = true;
          break;
        }
      }
      if (rawFound) break;
    }
    if (rawFound) continue;

    const container =
      pre.closest(
        '[class*="code"], [class*="Code"], [class*="highlight"], [class*="Highlight"], [class*="syntax"]'
      ) || pre.parentElement;
    if (container) {
      for (const btn of container.querySelectorAll('button, [role="button"]')) {
        for (const attr of rawAttrs) {
          const val = btn.getAttribute(attr);
          if (val && val.trim().length > 10) {
            pre.textContent = val;
            pre.setAttribute('data-sol-copied', '1');
            fixed++;
            rawFound = true;
            break;
          }
        }
        if (rawFound) break;
      }
    }
    if (rawFound) continue;

    const clone = pre.cloneNode(true);

    const lineNumSelectors = [
      '.line-number', '.linenumber', '.line-numbers', '.ln-num',
      '.hljs-ln-numbers', '.hljs-ln-n',
      '.rouge-gutter', '.gutter', '.gutter-cell',
      'td.line-numbers', 'td.gutter', 'td.hljs-ln-numbers',
      'td.blob-num', 'td.js-line-number',
      '[data-line-number]', '.code-line-number',
      '.react-syntax-highlighter-line-number',
      'span.linenumber',
    ];

    let removedAny = false;
    for (const sel of lineNumSelectors) {
      const els = clone.querySelectorAll(sel);
      if (els.length) {
        els.forEach((el) => el.remove());
        removedAny = true;
      }
    }

    if (!removedAny) {
      const spans = Array.from(clone.querySelectorAll('span'));
      const numSpans = spans.filter(
        (s) => s.children.length === 0 && /^\s*\d{1,5}\s*$/.test(s.textContent)
      );
      if (numSpans.length >= 5) {
        const nums = numSpans.map((s) => parseInt(s.textContent.trim(), 10));
        const sequential = nums.every((n, i) => i === 0 || n === nums[i - 1] + 1);
        if (sequential && nums[0] === 1) {
          numSpans.forEach((s) => s.remove());
          removedAny = true;
        }
      }
    }

    if (removedAny) {
      const walker = document.createTreeWalker(clone, NodeFilter.SHOW_TEXT);
      const parts = [];
      while (walker.nextNode()) parts.push(walker.currentNode.nodeValue);
      const clean = parts.join('');
      if (clean && clean.trim()) {
        pre.textContent = clean;
        pre.setAttribute('data-sol-copied', '1');
        fixed++;
      }
    }
  }
  return fixed;
}
"""


TRUNCATION_FIX_CSS = r"""
*,
*::before,
*::after {
  text-overflow: clip !important;
}

[class*="truncate"],
.truncate,
[class*="line-clamp"],
[style*="text-overflow"],
[style*="overflow: hidden"] {
  overflow: visible !important;
  text-overflow: clip !important;
  white-space: normal !important;
  max-width: none !important;
  -webkit-line-clamp: unset !important;
}
"""

TEXT_ATTRS = (
    "data-full-value",
    "data-value",
    "data-tooltip",
    "data-original-title",
    "title",
    "aria-label",
)

RECOGNIZED_BLOCKS = {
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "table", "pre", "ul", "ol", "blockquote", "hr"
}

# ---------------------------------------------------------------------------
# >>>  NEW: operation-name → short-slug mapping  <<<
# Sorted longest-first so the most specific alias wins.
# ---------------------------------------------------------------------------
OPERATION_ALIASES: dict[str, str] = {
    "root mean square normalization": "rmsnorm",
    "rms layer normalization": "rmsnorm",
    "rms normalization": "rmsnorm",
    "rmsnorm": "rmsnorm",
    "rms norm": "rmsnorm",
    "layer normalization": "layernorm",
    "layer norm": "layernorm",
    "layernorm": "layernorm",
    "batch normalization": "batchnorm",
    "batch norm": "batchnorm",
    "batchnorm": "batchnorm",
    "group normalization": "groupnorm",
    "group norm": "groupnorm",
    "instance normalization": "instancenorm",
    "instance norm": "instancenorm",
    "softmax with softcapping": "softmax_softcap",
    "log softmax": "log_softmax",
    "softmax": "softmax",
    "general matrix multiplication": "gemm",
    "batched matrix multiplication": "bmm",
    "matrix multiplication": "matmul",
    "matrix multiply": "matmul",
    "matmul": "matmul",
    "gemm": "gemm",
    "bmm": "bmm",
    "depthwise convolution": "dwconv",
    "pointwise convolution": "pwconv",
    "conv2d": "conv2d",
    "conv1d": "conv1d",
    "convolution": "conv",
    "grouped query attention": "gqa",
    "multi-head attention": "mha",
    "multi head attention": "mha",
    "flash attention": "flash_attn",
    "self-attention": "self_attn",
    "self attention": "self_attn",
    "cross attention": "cross_attn",
    "attention": "attn",
    "cross entropy loss": "ce_loss",
    "binary cross entropy": "bce",
    "cross entropy": "cross_entropy",
    "cross-entropy": "cross_entropy",
    "mixture of experts": "moe",
    "moe routing": "moe_routing",
    "moe": "moe",
    "dropout": "dropout",
    "leaky relu": "leaky_relu",
    "relu": "relu",
    "gelu": "gelu",
    "silu": "silu",
    "swiglu": "swiglu",
    "mish": "mish",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "elementwise": "elementwise",
    "element-wise": "elementwise",
    "element wise": "elementwise",
    "pointwise": "pointwise",
    "sum reduction": "sum_reduce",
    "mean reduction": "mean_reduce",
    "reduction": "reduce",
    "scatter add": "scatter_add",
    "scatter reduce": "scatter_reduce",
    "scatter": "scatter",
    "index select": "index_select",
    "index add": "index_add",
    "gather": "gather",
    "rotary positional embedding": "rope",
    "rotary embedding": "rope",
    "positional encoding": "pos_enc",
    "rope": "rope",
    "embedding": "embedding",
    "linear projection": "linear_proj",
    "linear": "linear",
    "fully connected": "fc",
    "adaptive average pooling": "adaptive_avgpool",
    "global average pooling": "gap",
    "average pooling": "avgpool",
    "max pooling": "maxpool",
    "avg pool": "avgpool",
    "max pool": "maxpool",
    "pooling": "pool",
    "fused multiply add": "fma",
    "fused add rmsnorm": "fused_add_rmsnorm",
    "fused residual norm": "fused_res_norm",
    "residual connection": "residual",
    "residual": "residual",
    "skip connection": "skip",
    "transpose": "transpose",
    "permute": "permute",
    "reshape": "reshape",
    "concatenation": "concat",
    "concatenate": "concat",
    "flatten": "flatten",
    "unfold": "unfold",
    "slice": "slice",
    "pad": "pad",
    "topk": "topk",
    "top-k": "topk",
    "top k": "topk",
    "sort": "sort",
    "argsort": "argsort",
    "cumulative sum": "cumsum",
    "cumsum": "cumsum",
    "prefix sum": "prefix_sum",
    "scan": "scan",
    "histogram": "histogram",
    "quantization": "quant",
    "dequantization": "dequant",
    "state space model": "ssm",
    "selective scan": "selective_scan",
    "mamba": "mamba",
    "backward": "bwd",
    "forward": "fwd",
    "gradient": "grad",
    "kv cache": "kv_cache",
    "paged attention": "paged_attn",
}

# ---------------------------------------------------------------------------
# >>>  NEW: parameter-name → short abbreviation  <<<
# ---------------------------------------------------------------------------
PARAM_ALIASES: dict[str, str] = {
    "hidden_size": "h",
    "hidden": "h",
    "batch_size": "b",
    "batch": "b",
    "seq_len": "s",
    "seq_length": "s",
    "sequence_length": "s",
    "sequence_len": "s",
    "num_heads": "nh",
    "n_heads": "nh",
    "nheads": "nh",
    "head_dim": "hd",
    "head_size": "hd",
    "d_model": "dm",
    "model_dim": "dm",
    "vocab_size": "v",
    "in_features": "inf",
    "out_features": "outf",
    "input_size": "in",
    "output_size": "out",
    "kernel_size": "k",
    "stride": "st",
    "padding": "pad",
    "groups": "g",
    "embed_dim": "ed",
    "embedding_dim": "ed",
    "num_layers": "nl",
    "n_layers": "nl",
    "channels": "c",
    "in_channels": "ic",
    "out_channels": "oc",
    "height": "ht",
    "width": "w",
    "dim": "d",
    "size": "sz",
    "length": "len",
    "depth": "dp",
    "num_elements": "n",
    "intermediate_size": "inter",
    "num_experts": "ne",
    "n_experts": "ne",
    "topk": "topk",
    "top_k": "topk",
    "n": "n",
    "m": "m",
    "k": "k",
}

# Parameters to exclude from the short filename (fixed / unimportant)
SKIP_PARAMS: set[str] = {
    "epsilon", "eps", "dtype", "device", "requires_grad",
    "training", "inplace", "bias", "affine",
}

# ---------------------------------------------------------------------------
# helpers – unchanged from original
# ---------------------------------------------------------------------------


def normalize_ws(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_code(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip("\n")


def clean_non_code_text(text: str) -> str:
    text = normalize_ws(text)
    text = re.sub(r"\\(?=[_\[\]])", "", text)
    return text


def md_escape_cell(text: str) -> str:
    text = clean_non_code_text(text)
    text = text.replace("|", r"\|")
    text = text.replace("\n", "<br>")
    return text


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out

def extract_kernel_name(page) -> str:
    """
    Extract the exact kernel name already present on the page.

    Looks for text like '#1 001_attention_softmax_dropout_value_matmul_backward'
    and returns '001_attention_softmax_dropout_value_matmul_backward'.
    Falls back to any 3-digit_snake_case token found on the page.
    """
    name = page.evaluate(r"""
        () => {
            const body = document.body;
            if (!body) return '';

            // ── helper: try a regex on a string, return first capture or '' ──
            const tryMatch = (text, re) => { const m = text.match(re); return m ? m[1] : ''; };

            // Pattern A: "#<id>  NNN_snake_name"
            const PAT_HASH = /#\s*\d+\s+(\d{3}_[a-zA-Z0-9_]+)/;
            // Pattern B: bare "NNN_word_word…" (at least two underscored segments)
            const PAT_BARE = /\b(\d{3}_[a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b/i;

            // Strategy 1 — headings (most reliable)
            for (const h of document.querySelectorAll('h1,h2,h3,h4,h5,h6')) {
                const t = (h.innerText || h.textContent || '').trim();
                const r = tryMatch(t, PAT_HASH) || tryMatch(t, PAT_BARE);
                if (r) return r;
            }

            // Strategy 2 — prominent class-named elements
            const classSelectors = [
                '[class*="title" i]', '[class*="name" i]',
                '[class*="header" i]', '[class*="kernel" i]',
                '[class*="problem" i]', '[class*="benchmark" i]',
            ];
            for (const sel of classSelectors) {
                try {
                    for (const el of document.querySelectorAll(sel)) {
                        const t = (el.innerText || el.textContent || '').trim();
                        const r = tryMatch(t, PAT_HASH) || tryMatch(t, PAT_BARE);
                        if (r) return r;
                    }
                } catch (_) {}
            }

            // Strategy 3 — full visible page text
            const full = body.innerText || '';
            return tryMatch(full, PAT_HASH) || tryMatch(full, PAT_BARE) || '';
        }
    """)
    return name.strip() if name else ""

def extract_pre_text(node: Tag) -> str:
    parts: list[str] = []
    for child in node.descendants:
        if isinstance(child, NavigableString):
            parts.append(str(child))
    return "".join(parts)


_LINE_NUM_RE = re.compile(
    r"""
    ^                 # start of line
    [ \t]*            # optional leading whitespace
    (\d{1,5})         # a 1-5 digit number (candidate line number)
    (?=\n|$)          # followed by newline or end-of-string
    """,
    re.MULTILINE | re.VERBOSE,
)


def strip_embedded_line_numbers(code: str) -> str:
    matches = list(_LINE_NUM_RE.finditer(code))
    if len(matches) < 5:
        return code
    nums = [int(m.group(1)) for m in matches]
    if nums[0] != 1:
        return code
    if not all(nums[i] == nums[i - 1] + 1 for i in range(1, len(nums))):
        return code
    lines = code.split("\n")
    offset = 0
    line_starts: dict[int, int] = {}
    for i, line in enumerate(lines):
        line_starts[offset] = i
        offset += len(line) + 1
    kill = {line_starts[m.start()] for m in matches if m.start() in line_starts}
    cleaned = [line for i, line in enumerate(lines) if i not in kill]
    return "\n".join(cleaned)


def best_text(node: Tag) -> str:
    candidates: list[str] = []
    visible = normalize_ws(node.get_text(" ", strip=True))
    if visible:
        candidates.append(visible)
    for attr in TEXT_ATTRS:
        value = node.get(attr)
        if value:
            candidates.append(normalize_ws(value))
    for desc in node.find_all(True):
        for attr in TEXT_ATTRS:
            value = desc.get(attr)
            if value:
                candidates.append(normalize_ws(value))
    candidates = dedupe_keep_order([c for c in candidates if c])
    if not candidates:
        return ""
    return max(candidates, key=len)


# ---------------------------------------------------------------------------
# >>>  NEW: extract description & problem index from page  <<<
# ---------------------------------------------------------------------------

def extract_description(page) -> str:
    """
    Pull the 'Description' section text out of the kernel page.
    Handles label+value pairs, concatenated text, and various DOM layouts.
    """
    raw = page.evaluate(r"""
        () => {
            const body = document.body;
            if (!body) return '';

            // ── Strategy 1: label element whose text is exactly "Description",
            //    then grab the immediately following sibling / parent-sibling.
            const walk = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walk.nextNode())) {
                if (/^\s*Description\s*$/i.test(node.textContent)) {
                    const parent = node.parentElement;
                    if (!parent) continue;
                    // <dt>Description</dt><dd>…</dd>  or  <th>…</th><td>…</td>
                    const sib = parent.nextElementSibling;
                    if (sib) {
                        const t = (sib.innerText || '').trim();
                        if (t.length > 5) return t;
                    }
                    // try one level up
                    const psib = parent.parentElement
                                 ? parent.parentElement.nextElementSibling
                                 : null;
                    if (psib) {
                        const t = (psib.innerText || '').trim();
                        if (t.length > 5) return t;
                    }
                }
            }

            // ── Strategy 2: element whose innerText starts with "Description"
            //    (handles concatenated label + value).
            const candidates = body.querySelectorAll(
                'p, div, span, td, dd, section, [class*="desc" i]'
            );
            for (const el of candidates) {
                const t = (el.innerText || '').trim();
                if (/^Description\s*/i.test(t) && t.length > 15) {
                    return t.replace(/^Description\s*[:\-–]?\s*/i, '');
                }
            }

            // ── Strategy 3: regex on full page text
            const full = (body.innerText || '');
            const m = full.match(
                /Description\s*[:\-–]?\s*\n?\s*(.+?)(?=\n\s*(?:\n|[A-Z][a-z]+\s*[:\-–]))/s
            );
            if (m && m[1].trim().length > 5) return m[1].trim();

            return '';
        }
    """)
    return clean_non_code_text(raw) if raw else ""


def extract_problem_index(page) -> str:
    """
    Try to find a visible problem / benchmark index on the page
    (e.g. "Problem 026", "#026", "Kernel 26", a breadcrumb number, etc.).
    Returns the raw numeric string or '' if nothing found.
    """
    idx = page.evaluate(r"""
        () => {
            const body = document.body;
            if (!body) return '';
            const text = body.innerText || '';

            // ── labelled patterns ──
            const patterns = [
                /(?:Problem|Benchmark|Test[\s-]*Case|Kernel)\s*(?:Number|#|No\.?|Index|ID)?\s*[:\-–]?\s*(\d{1,4})/i,
                /(?:Index|Number|No\.?|ID)\s*[:\-–]?\s*(\d{1,4})/i,
                /#\s*(\d{1,4})\b/,
            ];
            for (const pat of patterns) {
                const m = text.match(pat);
                if (m) return m[1];
            }

            // ── data-attributes on the page container ──
            const meta = document.querySelector(
                '[data-problem-id], [data-kernel-id], [data-index], [data-problem]'
            );
            if (meta) {
                for (const attr of ['data-problem-id','data-kernel-id','data-index','data-problem']) {
                    const v = meta.getAttribute(attr);
                    if (v && /^\d{1,4}$/.test(v.trim())) return v.trim();
                }
            }

            // ── breadcrumb / nav link text with a bare number ──
            const crumbs = document.querySelectorAll(
                'nav a, [class*="breadcrumb"] a, [aria-label="breadcrumb"] a'
            );
            for (const a of crumbs) {
                const t = (a.innerText || '').trim();
                if (/^\d{1,4}$/.test(t)) return t;
            }

            return '';
        }
    """)
    return idx.strip()


# ---------------------------------------------------------------------------
# >>>  NEW: generate a short problem name from the description  <<<
# ---------------------------------------------------------------------------

def generate_short_name(description: str) -> str:
    """
    Turn a natural-language description into a terse filename slug.

    Examples
    --------
    "Root Mean Square Normalization with hidden_size=7168. …"
        → "rmsnorm_h7168"
    "Softmax with seq_len=2048, num_heads=32. …"
        → "softmax_s2048_nh32"
    """
    if not description:
        return "unknown"

    desc_lower = description.lower().strip()

    # --- find the best-matching operation alias (longest match wins) ---
    op_name = ""
    best_len = 0
    for full, short in OPERATION_ALIASES.items():
        if full in desc_lower and len(full) > best_len:
            op_name = short
            best_len = len(full)

    # fallback: first few meaningful words
    if not op_name:
        first_part = description.split(".")[0]
        first_part = re.split(r"\s+with\s+", first_part, maxsplit=1)[0]
        words = re.findall(r"[a-zA-Z]+", first_part)
        stop = {
            "with", "and", "or", "the", "a", "an", "for", "from", "of",
            "in", "on", "at", "to", "is", "are", "was", "were",
            "captured", "using", "based", "kernel", "operation",
        }
        meaningful = [w.lower() for w in words if w.lower() not in stop]
        op_name = "_".join(meaningful[:3]) if meaningful else "unknown"

    # --- extract parameter=value pairs ---
    params = re.findall(
        r"(\w+)\s*=\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        description,
    )

    parts: list[str] = []
    for pname, pval in params:
        if pname.lower() in SKIP_PARAMS:
            continue
        abbrev = PARAM_ALIASES.get(pname.lower(), pname.lower()[:3])
        # tidy the numeric value
        try:
            if "." not in pval and "e" not in pval.lower():
                pval = str(int(pval))
            else:
                f = float(pval)
                if f == int(f) and "e" not in pval.lower():
                    pval = str(int(f))
        except ValueError:
            pass
        parts.append(f"{abbrev}{pval}")

    if parts:
        return f"{op_name}_{'_'.join(parts)}"
    return op_name


def sanitize_filename(name: str) -> str:
    """Replace anything not alphanumeric / underscore / hyphen."""
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# ---------------------------------------------------------------------------
# page-level copy-button helpers  (unchanged)
# ---------------------------------------------------------------------------

def mark_copyable_code_blocks(page) -> int:
    return page.evaluate(
        r"""
        () => {
          const labels = (el) => [
            el.innerText,
            el.textContent,
            el.getAttribute("aria-label"),
            el.getAttribute("title"),
            el.getAttribute("data-tooltip"),
            el.getAttribute("data-original-title"),
          ].filter(Boolean).join(" ");

          const isCopyButton = (el) => {
            if (/\bcopy\b|\bcopied\b/i.test(labels(el))) return true;
            const cls = (el.className || "") + " " + (el.id || "");
            if (/\bcopy\b|\bclipboard\b/i.test(cls)) return true;
            const dataAttrs = [
              "data-action", "data-testid", "data-cy", "data-test",
              "data-clipboard-target", "data-clipboard-text",
            ];
            for (const attr of dataAttrs) {
              if (/copy|clipboard/i.test(el.getAttribute(attr) || "")) return true;
            }
            const svg = el.querySelector("svg");
            if (svg) {
              const svgMeta = [
                svg.getAttribute("aria-label"),
                svg.getAttribute("data-icon"),
                svg.getAttribute("data-testid"),
                svg.getAttribute("class"),
                svg.id,
              ].filter(Boolean).join(" ");
              if (/copy|clipboard|duplicate|content.copy/i.test(svgMeta)) return true;
              const use = svg.querySelector("use");
              if (use) {
                const href = use.getAttribute("href") || use.getAttribute("xlink:href") || "";
                if (/copy|clipboard/i.test(href)) return true;
              }
            }
            if (el.querySelector('[class*="copy"], [class*="Copy"], [class*="clipboard"]')) {
              return true;
            }
            return false;
          };

          document.querySelectorAll("[data-sol-copy-btn-idx]").forEach((el) => {
            el.removeAttribute("data-sol-copy-btn-idx");
          });
          document
            .querySelectorAll(
              "[data-sol-copy-idx], [data-sol-original-lang], [data-sol-lang], [data-sol-copied]"
            )
            .forEach((el) => {
              el.removeAttribute("data-sol-copy-idx");
              el.removeAttribute("data-sol-original-lang");
              el.removeAttribute("data-sol-lang");
              el.removeAttribute("data-sol-copied");
            });

          let idx = 0;
          const seen = new Set();
          const candidates = Array.from(
            document.querySelectorAll(
              "button, [role='button'], [data-clipboard-target], [data-clipboard-text]"
            )
          );

          for (const btn of candidates) {
            if (!isCopyButton(btn)) continue;
            let cur = btn;
            let host = null;
            for (let depth = 0; depth < 15 && cur; depth++, cur = cur.parentElement) {
              if (cur.querySelector("pre")) {
                host = cur;
                break;
              }
            }
            if (!host) continue;
            const pre = btn.closest("pre") || host.querySelector("pre");
            if (!pre || seen.has(pre)) continue;
            const code = pre.querySelector("code");
            const classes = [
              ...(code ? Array.from(code.classList) : []),
              ...Array.from(pre.classList || []),
            ];
            let lang = "";
            for (const cls of classes) {
              if (cls.startsWith("language-")) {
                lang = cls.slice("language-".length);
                break;
              }
            }
            pre.setAttribute("data-sol-copy-idx", String(idx));
            if (lang) {
              pre.setAttribute("data-sol-original-lang", lang);
            }
            btn.setAttribute("data-sol-copy-btn-idx", String(idx));
            seen.add(pre);
            idx += 1;
          }
          return idx;
        }
        """
    )


def wait_for_copied_text(page, timeout_ms: int = 2000) -> str:
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        text = page.evaluate("() => window.__sol_last_copied_text || ''")
        text = normalize_code(text)
        if text:
            return text
        page.wait_for_timeout(50)
    return ""


def replace_code_blocks_using_copy_buttons(page) -> int:
    total = mark_copyable_code_blocks(page)
    copied = 0
    for idx in range(total):
        pre = page.locator(f'[data-sol-copy-idx="{idx}"]').first
        btn = page.locator(f'[data-sol-copy-btn-idx="{idx}"]').first
        try:
            pre.scroll_into_view_if_needed()
        except Exception:
            pass
        try:
            pre.hover()
        except Exception:
            pass
        page.evaluate("() => { window.__sol_last_copied_text = ''; }")
        clicked = False
        for force in (False, True):
            try:
                btn.click(timeout=4_000, force=force)
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            continue
        text = wait_for_copied_text(page, timeout_ms=2000)
        if not text:
            continue
        page.evaluate(
            r"""
            ({idx, text}) => {
              const pre = document.querySelector(`[data-sol-copy-idx="${idx}"]`);
              if (!pre) return;
              const lang = pre.getAttribute("data-sol-original-lang") || "";
              pre.textContent = text;
              if (lang) {
                pre.setAttribute("data-sol-lang", lang);
              }
              pre.setAttribute("data-sol-copied", "1");
            }
            """,
            {"idx": idx, "text": text},
        )
        copied += 1
    return copied


# ---------------------------------------------------------------------------
# HTML → Markdown conversion  (unchanged)
# ---------------------------------------------------------------------------

def choose_main_html(page) -> str:
    html = page.evaluate(
        """
        () => {
          const selectors = [
            "main",
            "article",
            "[role='main']",
            ".prose",
            ".markdown-body",
            ".content",
            ".container"
          ];
          const candidates = [];
          const seen = new Set();
          for (const sel of selectors) {
            for (const el of document.querySelectorAll(sel)) {
              if (!seen.has(el)) {
                seen.add(el);
                candidates.push(el);
              }
            }
          }
          if (!candidates.length) {
            candidates.push(document.body);
          }
          const score = (el) => {
            const textLen = (el.innerText || "").trim().length;
            const tables = el.querySelectorAll("table, [role='table']").length;
            const pres = el.querySelectorAll("pre").length;
            const headings = el.querySelectorAll("h1,h2,h3,h4").length;
            return textLen + tables * 5000 + pres * 3000 + headings * 1000;
          };
          candidates.sort((a, b) => score(b) - score(a));
          return candidates[0].outerHTML;
        }
        """
    )
    return html


def iter_blocks(node: Tag):
    for child in node.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in {"script", "style", "noscript", "svg", "nav", "aside", "footer", "button"}:
            continue
        if child.name in RECOGNIZED_BLOCKS or child.get("role") == "table":
            yield child
        else:
            yield from iter_blocks(child)


def table_to_md(table: Tag) -> str:
    rows: list[list[str]] = []
    trs = table.select("tr")
    for tr in trs:
        direct_cells = [c for c in tr.children if isinstance(c, Tag) and c.name in {"th", "td"}]
        if not direct_cells:
            direct_cells = tr.find_all(["th", "td"])
        row: list[str] = []
        for cell in direct_cells:
            text = best_text(cell)
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
            except Exception:
                colspan = 1
            row.extend([md_escape_cell(text)] * colspan)
        if any(cell.strip() for cell in row):
            rows.append(row)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def list_to_md(node: Tag) -> str:
    ordered = node.name == "ol"
    items = []
    for i, li in enumerate(node.find_all("li", recursive=False), start=1):
        text = clean_non_code_text(li.get_text(" ", strip=True))
        if not text:
            continue
        prefix = f"{i}. " if ordered else "- "
        items.append(prefix + text)
    return "\n".join(items)


def blockquote_to_md(node: Tag) -> str:
    text = clean_non_code_text(node.get_text("\n", strip=True))
    if not text:
        return ""
    return "\n".join("> " + line if line else ">" for line in text.splitlines())


def pre_to_md(node: Tag) -> str:
    if node.get("data-sol-copied") == "1":
        raw = node.get_text()
    else:
        raw = extract_pre_text(node)
    code = normalize_code(raw)
    if not code:
        return ""
    code = strip_embedded_line_numbers(code)
    code = normalize_code(code)
    if not code:
        return ""
    lang = node.get("data-sol-lang", "")
    if not lang:
        code_tag = node.find("code")
        if code_tag:
            for cls in code_tag.get("class", []):
                if cls.startswith("language-"):
                    lang = cls.split("-", 1)[1]
                    break
    if not lang and ("import torch" in code or "def run(" in code):
        lang = "python"
    return f"```{lang}\n{code}\n```"


def append_part(parts: list[str], part: str) -> None:
    part = part.strip()
    if not part:
        return
    if parts and parts[-1] == part:
        return
    parts.append(part)


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "aside", "footer", "button"]):
        tag.decompose()
    for tag in soup.select("[hidden], [aria-hidden='true']"):
        tag.decompose()
    root = soup.body or soup
    parts: list[str] = []
    for block in iter_blocks(root):
        if block.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = clean_non_code_text(best_text(block))
            if not text:
                continue
            level = int(block.name[1])
            if level == 1:
                append_part(parts, text)
            else:
                append_part(parts, f"{'#' * min(level, 6)} {text}")
        elif block.name == "p":
            text = clean_non_code_text(best_text(block))
            if text:
                append_part(parts, text)
        elif block.name == "table" or block.get("role") == "table":
            md = table_to_md(block)
            if md:
                append_part(parts, md)
        elif block.name == "pre":
            md = pre_to_md(block)
            if md:
                append_part(parts, md)
        elif block.name in {"ul", "ol"}:
            md = list_to_md(block)
            if md:
                append_part(parts, md)
        elif block.name == "blockquote":
            md = blockquote_to_md(block)
            if md:
                append_part(parts, md)
        elif block.name == "hr":
            append_part(parts, "---")
    text = "\n\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def fetch_kernel_markdown(page, kernel_id: int) -> dict:
    """
    Returns
    -------
    dict  with keys  ``markdown``, ``description``, ``problem_index``, ``kernel_name``
    """
    url = BASE_URL.format(kernel_id)

    response = page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    if response is not None and response.status >= 400:
        raise RuntimeError(f"HTTP {response.status} for kernel {kernel_id}")

    try:
        page.wait_for_load_state("networkidle", timeout=8_000)
    except PlaywrightTimeoutError:
        pass

    try:
        page.wait_for_selector("h1, h2, table, pre", timeout=15_000)
    except PlaywrightTimeoutError:
        pass

    page.add_style_tag(content=TRUNCATION_FIX_CSS)
    page.wait_for_timeout(300)

    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(250)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(250)

    # ── extract metadata BEFORE any DOM surgery ──
    description = extract_description(page)
    problem_index = extract_problem_index(page)
    kernel_name = extract_kernel_name(page)          # ← NEW

    # ── PRIMARY: click copy buttons and capture clipboard text ──
    replace_code_blocks_using_copy_buttons(page)

    # ── SECONDARY: DOM-level cleanup ──
    page.evaluate(CLEAN_UNCOPIED_CODE_JS)

    html = choose_main_html(page)
    if "<table" not in html and "<pre" not in html:
        html = page.content()

    md = html_to_markdown(html)

    return {
        "markdown": md,
        "description": description,
        "problem_index": problem_index,
        "kernel_name": kernel_name,                  # ← NEW
    }

# ---------------------------------------------------------------------------
# >>>  MODIFIED: crawl – new filename scheme + description in .md  <<<
# ---------------------------------------------------------------------------

def crawl(start: int, end: int, out_dir: Path, sleep_s: float, headful: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        context = browser.new_context(
            viewport={"width": 1720, "height": 1400},
            bypass_csp=True,
            java_script_enabled=True,
        )
        context.add_init_script(script=CLIPBOARD_CAPTURE_INIT_SCRIPT)
        try:
            context.grant_permissions(
                ["clipboard-read", "clipboard-write"],
                origin="https://research.nvidia.com",
            )
        except Exception:
            pass

        page = context.new_page()

        for kernel_id in range(start, end + 1):
            try:
                result = fetch_kernel_markdown(page, kernel_id)
                md = result["markdown"]
                description = result["description"]
                problem_index = result["problem_index"]
                kernel_name = result["kernel_name"]          # ← NEW

                if not md or len(md.splitlines()) < 5:
                    print(f"[WARN] kernel {kernel_id}: extracted content looks too small, skipping")
                    continue

                # ── build the filename ──
                if kernel_name:
                    # Use the exact name already on the page
                    stem = sanitize_filename(f"problem_{kernel_name}")
                else:
                    # Fallback: auto-generate from description
                    short_name = generate_short_name(description)
                    idx_part = problem_index.zfill(3) if problem_index else "000"
                    stem = sanitize_filename(
                        f"problem_{kernel_id}_{idx_part}_{short_name}"
                    )
                filename = f"{stem}.md"

                # ── prepend description to the markdown ──
                header_lines: list[str] = []
                if description:
                    header_lines.append(f"## Description\n\n{description}")
                header_lines.append("")
                final_md = "\n".join(header_lines) + md

                out_file = out_dir / filename
                out_file.write_text(final_md, encoding="utf-8")
                print(f"[OK] kernel {kernel_id:03d} -> {out_file.name}")

                if sleep_s > 0:
                    time.sleep(sleep_s)

            except Exception as e:
                print(f"[ERROR] kernel {kernel_id:03d}: {e}")
                
        context.close()
        browser.close()


def main():
    parser = argparse.ArgumentParser(
        description="Crawl SOL-ExecBench kernel pages into markdown files."
    )
    parser.add_argument("--start", type=int, default=1, help="Start kernel id")
    parser.add_argument("--end", type=int, default=235, help="End kernel id")
    parser.add_argument("--out", type=Path, default=Path("sol_execbench_md"),
                        help="Output directory")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Delay between pages in seconds")
    parser.add_argument("--headful", action="store_true",
                        help="Run browser with visible UI")
    args = parser.parse_args()

    crawl(
        start=args.start,
        end=args.end,
        out_dir=args.out,
        sleep_s=args.sleep,
        headful=args.headful,
    )


if __name__ == "__main__":
    main()