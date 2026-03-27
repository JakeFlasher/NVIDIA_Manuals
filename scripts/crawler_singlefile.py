#!/usr/bin/env python3
"""
NVIDIA Documentation Crawler & Markdown Converter
===================================================
Generic crawler for any Sphinx-based NVIDIA documentation site.
Crawls all content and saves each section as an individual Markdown file
with formatting matching the "Copy as Markdown" Chrome extension style.

Supports sites such as:
  - PTX ISA:                https://docs.nvidia.com/cuda/parallel-thread-execution/
  - CUDA Best Practices:    https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
  - CUDA C Programming:     https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  - cuBLAS, cuDNN, etc.

Requirements:
    pip install requests beautifulsoup4 lxml

Usage:
    python nvidia_doc_crawler.py <URL>
    python nvidia_doc_crawler.py <URL> -o my_docs --split-depth 3
    python nvidia_doc_crawler.py <URL> --single-file
    python nvidia_doc_crawler.py <URL> --download-images

Examples:
    python nvidia_doc_crawler.py https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html
    python nvidia_doc_crawler.py https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html
    python nvidia_doc_crawler.py https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ -o best_practices
"""

import os
import re
import sys
import time
import logging
import argparse
import urllib.parse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Generator
from dataclasses import dataclass, field
from collections import OrderedDict

try:
    import requests
except ImportError:
    sys.exit("Error: 'requests' is required.\n  pip install requests")

try:
    from bs4 import BeautifulSoup, Tag, NavigableString, Comment
except ImportError:
    sys.exit("Error: 'beautifulsoup4' is required.\n  pip install beautifulsoup4")

try:
    import lxml  # noqa: F401
    HTML_PARSER = "lxml"
except ImportError:
    HTML_PARSER = "html.parser"


# =============================================================================
# Constants
# =============================================================================
DEFAULT_DELAY = 0.8          # seconds between HTTP requests
MAX_RETRIES   = 3
RETRY_WAIT    = 5.0          # base wait on retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nvidia_doc_crawler")


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class SectionInfo:
    """Metadata about one <section> extracted from HTML."""
    id: str
    title: str
    section_number: str          # e.g. "9.7.8"
    heading_level: int           # 1-6
    element: Tag                 # bs4 element
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0               # nesting depth (0 = top-level chapter)


# =============================================================================
# URL Resolver
# =============================================================================
class DocURLResolver:
    """
    Given any NVIDIA docs URL, resolve the canonical base URL, the TOC URL,
    and a human-friendly slug for naming the output directory.

    Supports inputs such as:
        .../parallel-thread-execution/contents.html
        .../cuda-c-best-practices-guide/
        .../cuda-c-best-practices-guide/index.html
        .../cuda-c-best-practices-guide/index.html#some-anchor
    """

    @staticmethod
    def resolve(url: str) -> Tuple[str, str, str]:
        """
        Returns
        -------
        base_url : str   – root of the doc, always ends with '/'
        toc_url  : str   – best-guess URL for the table-of-contents page
        slug     : str   – short name derived from the path
        """
        parsed = urllib.parse.urlparse(url)
        # Strip fragment
        path = parsed.path.rstrip("/")

        if path.endswith("/contents.html") or path.endswith("/contents"):
            base_path = path.rsplit("/", 1)[0] + "/"
            toc_path  = base_path + "contents.html"
        elif path.endswith("/index.html") or path.endswith("/index"):
            base_path = path.rsplit("/", 1)[0] + "/"
            toc_path  = base_path + "contents.html"
        elif path.endswith(".html"):
            base_path = path.rsplit("/", 1)[0] + "/"
            toc_path  = base_path + "contents.html"
        else:
            base_path = path + "/" if not path.endswith("/") else path
            toc_path  = base_path + "contents.html"

        base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
        toc_url  = f"{parsed.scheme}://{parsed.netloc}{toc_path}"

        # Slug from last path component
        parts = base_path.strip("/").split("/")
        slug  = parts[-1] if parts else "nvidia-docs"

        return base_url, toc_url, slug

    @staticmethod
    def verify_toc(toc_url: str, base_url: str,
                   session: requests.Session) -> str:
        """
        Check whether *toc_url* actually exists.  If not, try common
        fall-backs and return the first that responds with 200.
        """
        candidates = [
            toc_url,
            base_url + "contents.html",
            base_url + "index.html",
            base_url,
        ]
        # deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        for url in unique:
            try:
                r = session.head(url, timeout=15, allow_redirects=True)
                if r.status_code == 200:
                    return url
            except requests.RequestException:
                continue
        # If nothing works, return the original guess
        return toc_url


# =============================================================================
# HTML Pre-processor  (expand hidden content, clean code blocks)
# =============================================================================
class HTMLPreprocessor:
    """
    In-place mutations on a BeautifulSoup tree to ensure *all* content is
    visible to the Markdown converter — hidden tabs, collapsed details,
    display:none blocks, line-number gutters, copy buttons, etc.
    """

    @staticmethod
    def preprocess(soup: BeautifulSoup) -> BeautifulSoup:
        # ---- 1. Remove copy-to-clipboard buttons -------------------------
        for sel in (
            {"class_": lambda c: c and any("copy" in x.lower() for x in
                                            (c if isinstance(c, list) else [c]))},
            {"attrs": {"data-clipboard-target": True}},
            {"class_": "copybtn"},
            {"class_": "copybutton"},
        ):
            for btn in soup.find_all("button", **sel):
                btn.decompose()

        # ---- 2. Code blocks with line-number tables ----------------------
        for table in soup.find_all("table", class_="highlighttable"):
            code_cell = table.find("td", class_="code")
            if code_cell:
                hl = code_cell.find(
                    "div", class_=lambda c: c and "highlight" in
                    (" ".join(c) if isinstance(c, list) else c))
                if hl:
                    table.replace_with(hl)
        # standalone line-number gutters
        for div in soup.find_all("div", class_="linenodiv"):
            div.decompose()
        for td in soup.find_all("td", class_="linenos"):
            td.decompose()

        # ---- 3. Reveal display:none elements -----------------------------
        for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            el["style"] = re.sub(
                r"display\s*:\s*none\s*;?\s*", "", el.get("style", ""),
                flags=re.I,
            )

        # ---- 4. Remove hidden attribute ----------------------------------
        for el in soup.find_all(attrs={"hidden": True}):
            del el["hidden"]

        # ---- 5. Expand tabbed content (sphinx-tabs) ----------------------
        for panel in soup.find_all(
            "div", class_=lambda c: c and (
                "sphinx-tabs-panel" in (c if isinstance(c, list) else [c])
            )
        ):
            panel.attrs.pop("style", None)
            panel["aria-hidden"] = "false"

        # sphinx-design tabs
        for panel in soup.find_all("div", class_="sd-tab-content"):
            panel.attrs.pop("style", None)
            cls = panel.get("class", [])
            panel["class"] = [c for c in cls if c != "d-none"]

        # ---- 6. Expand Bootstrap-style collapse --------------------------
        for el in soup.find_all(class_="collapse"):
            cls = el.get("class", [])
            el["class"] = [c for c in cls
                           if c not in ("collapse", "collapsed")]

        # ---- 7. Remove aria-hidden="true" on content divs ---------------
        for el in soup.find_all(attrs={"aria-hidden": "true"}):
            tag = el.name or ""
            if tag in ("div", "section", "article", "pre", "code",
                       "table", "ul", "ol", "dl", "p", "span"):
                el["aria-hidden"] = "false"

        return soup


# =============================================================================
# HTML → Markdown Converter
# =============================================================================
class MarkdownConverter:
    """
    Comprehensive HTML-to-Markdown converter for Sphinx / NVIDIA docs.

    Produces output comparable to what the "Copy as Markdown" Chrome
    extension generates, covering:
    headings, paragraphs, links, images, figures, code blocks (with
    language tags), tables, nested lists (ul/ol/dl), admonitions,
    math blocks, blockquotes, footnotes, version directives, field
    lists, rubrics, line blocks, tabbed content, API descriptions,
    and more.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._img_counter = 0
        self._in_pre   = False
        self._in_table = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(self, element: Tag) -> str:
        self._img_counter = 0
        raw = self._el(element)
        return self._cleanup(raw)

    def convert_section_body(self, section_el: Tag,
                             include_children: bool = False) -> str:
        self._img_counter = 0
        parts: list[str] = []
        for child in section_el.children:
            if (not include_children
                    and isinstance(child, Tag)
                    and child.name == "section"):
                continue
            parts.append(self._el(child))
        return self._cleanup("".join(parts))

    # ------------------------------------------------------------------
    # Core recursive dispatcher
    # ------------------------------------------------------------------
    def _el(self, node, depth: int = 0) -> str:
        if node is None:
            return ""
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return str(node) if not self._in_pre else str(node)
        if not isinstance(node, Tag):
            return ""
        if node.name in ("script", "style", "link", "meta",
                         "noscript", "nav", "footer", "template"):
            return ""
        handler = getattr(self, f"_t_{node.name}", None)
        return handler(node, depth) if handler else self._children(node, depth)

    def _children(self, el: Tag, depth: int = 0) -> str:
        return "".join(self._el(c, depth) for c in el.children)

    def _inline(self, el) -> str:
        if isinstance(el, NavigableString):
            return str(el)
        if not isinstance(el, Tag):
            return ""
        parts: list[str] = []
        for c in el.children:
            if isinstance(c, NavigableString):
                parts.append(str(c))
            elif isinstance(c, Tag):
                h = getattr(self, f"_t_{c.name}", None)
                parts.append(h(c, 0) if h else self._inline(c))
        return "".join(parts)

    # ------------------------------------------------------------------
    # Headings  h1..h6
    # ------------------------------------------------------------------
    def _t_h1(self, e, d): return self._heading(e, 1)
    def _t_h2(self, e, d): return self._heading(e, 2)
    def _t_h3(self, e, d): return self._heading(e, 3)
    def _t_h4(self, e, d): return self._heading(e, 4)
    def _t_h5(self, e, d): return self._heading(e, 5)
    def _t_h6(self, e, d): return self._heading(e, 6)

    def _heading(self, el: Tag, level: int) -> str:
        text = self._heading_text(el)
        anchor = self._heading_anchor(el)
        sec_url = f"{self.base_url}#{anchor}" if anchor else ""

        pl = ""
        hl = el.find("a", class_="headerlink")
        if hl:
            href = self._abs(hl.get("href", ""))
            pl = f'[]({href} "Permalink to this headline")'

        escaped = re.sub(r"(\d)\.", r"\1\\.", text)
        prefix = "#" * level
        if sec_url:
            return f"\n{prefix} [{text}]({sec_url}){pl}\n\n"
        return f"\n{prefix} {escaped}{pl}\n\n"

    def _heading_text(self, el: Tag) -> str:
        parts: list[str] = []
        for c in el.children:
            if isinstance(c, Tag) and "headerlink" in c.get("class", []):
                continue
            parts.append(c.get_text() if isinstance(c, Tag) else str(c))
        return "".join(parts).strip()

    def _heading_anchor(self, el: Tag) -> str:
        if el.get("id"):
            return el["id"]
        p = el.parent
        if (p and isinstance(p, Tag)
                and p.name in ("section", "div") and p.get("id")):
            return p["id"]
        hl = el.find("a", class_="headerlink")
        if hl and hl.get("href", "").startswith("#"):
            return hl["href"][1:]
        return ""

    # ------------------------------------------------------------------
    # Block elements
    # ------------------------------------------------------------------
    def _t_p(self, el, d):
        cls = el.get("class", [])
        # Rubric: a sub-heading that doesn't appear in the TOC
        if "rubric" in cls:
            t = self._inline(el).strip()
            return f"\n**{t}**\n\n" if t else ""
        t = self._inline(el).strip()
        return f"{t}\n\n" if t else ""

    def _t_div(self, el, d):
        cls = el.get("class", [])
        cls_str = " ".join(cls) if isinstance(cls, list) else cls

        # --- admonitions / callouts ---
        if any(c in cls for c in ("admonition", "note", "warning", "tip",
                                   "important", "caution", "danger",
                                   "error", "hint", "seealso", "todo")):
            return self._admonition(el, d)

        # --- code blocks (highlight wrapper) ---
        if any(c.startswith("highlight") for c in cls):
            pre = el.find("pre")
            return self._t_pre(pre, d) if pre else self._children(el, d)

        # --- literal-block-wrapper (code block with caption) ---
        if "literal-block-wrapper" in cls:
            return self._literal_block_wrapper(el, d)

        # --- figures ---
        if any(c in cls for c in ("figure", "align-default",
                                   "align-center", "align-left",
                                   "align-right")):
            return self._figure(el, d)

        # --- math ---
        if "math" in cls:
            return f"\n$$\n{el.get_text().strip()}\n$$\n\n"

        # --- version directives ---
        if any(c in cls for c in ("versionadded", "versionchanged",
                                   "deprecated", "versionremoved")):
            return self._version_directive(el, d)

        # --- sphinx-tabs container ---
        if "sphinx-tabs" in cls:
            return self._sphinx_tabs(el, d)
        if "sd-tab-set" in cls:
            return self._sd_tab_set(el, d)

        # --- topic / sidebar / contents ---
        if any(c in cls for c in ("topic", "sidebar", "contents")):
            return self._admonition(el, d)

        # --- container directive (may wrap anything) ---
        if "container" in cls:
            return self._children(el, d)

        # --- line-block ---
        if "line-block" in cls:
            return self._line_block(el, d)

        return self._children(el, d)

    def _t_section(self, el, d):   return self._children(el, d)
    def _t_article(self, el, d):   return self._children(el, d)
    def _t_main(self, el, d):      return self._children(el, d)
    def _t_header(self, el, d):    return ""
    def _t_hr(self, el, d):        return "\n---\n\n"
    def _t_br(self, el, d):        return "<br>" if self._in_table else "  \n"

    def _t_blockquote(self, el, d):
        body = self._children(el, d).strip()
        return "\n".join(f"> {ln}" for ln in body.split("\n")) + "\n\n"

    # --- <aside> (modern Sphinx footnotes / notes) ---
    def _t_aside(self, el: Tag, d) -> str:
        cls = el.get("class", [])
        role = el.get("role", "")
        if "footnote" in cls or role == "note":
            label = el.find("span", class_="label")
            label_txt = label.get_text().strip() if label else ""
            body_parts: list[str] = []
            for child in el.children:
                if isinstance(child, Tag) and child == label:
                    continue
                body_parts.append(self._el(child, d))
            body = "".join(body_parts).strip()
            if label_txt:
                return f"\n[^{label_txt}]: {body}\n\n"
            return f"\n> {body}\n\n"
        if any(c in cls for c in ("note", "warning", "tip", "important",
                                   "caution", "danger", "error", "hint",
                                   "seealso", "todo", "admonition")):
            return self._admonition(el, d)
        return self._children(el, d)

    # ------------------------------------------------------------------
    # Code
    # ------------------------------------------------------------------
    def _t_pre(self, el: Tag, d) -> str:
        self._in_pre = True
        code_el = el.find("code")
        if code_el:
            lang = self._detect_lang(code_el) or self._detect_lang(el)
            text = code_el.get_text()
        else:
            lang = self._detect_lang(el)
            text = el.get_text()
        lines = text.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        text = "\n".join(lines)
        self._in_pre = False
        return f"\n```{lang}\n{text}\n```\n\n"

    def _t_code(self, el, d):
        if self._in_pre:
            return el.get_text()
        t = el.get_text()
        if "\n" in t and len(t) > 60:
            lang = self._detect_lang(el)
            return f"\n```{lang}\n{t.strip()}\n```\n\n"
        return f"``{t}``" if "`" in t else f"`{t}`"

    def _t_kbd(self, el, d):  return f"`{el.get_text()}`"
    def _t_samp(self, el, d): return f"`{el.get_text()}`"
    def _t_var(self, el, d):  return f"*{el.get_text()}*"
    def _t_tt(self, el, d):   return f"`{el.get_text()}`"

    def _detect_lang(self, el: Tag) -> str:
        for cls in el.get("class", []):
            if cls.startswith("language-"):
                return cls[9:]
            if cls.startswith("highlight-"):
                return cls[10:]
        p = el.parent
        while p and isinstance(p, Tag):
            for cls in p.get("class", []):
                if cls.startswith("highlight-"):
                    return cls[10:]
                if cls.startswith("language-"):
                    return cls[9:]
            p = p.parent
        return ""

    # --- literal-block-wrapper (code block with a caption) ---
    def _literal_block_wrapper(self, el: Tag, d) -> str:
        caption_el = el.find("div", class_="code-block-caption") or \
                     el.find("span", class_="caption-text")
        caption = ""
        if caption_el:
            caption = self._inline(caption_el).strip()
            caption = re.sub(r'\[\]\([^)]*"Permalink[^"]*"\)', "", caption)

        pre = el.find("pre")
        code_md = self._t_pre(pre, d) if pre else self._children(el, d)

        if caption:
            return f"*{caption}*\n{code_md}"
        return code_md

    # ------------------------------------------------------------------
    # Lists
    # ------------------------------------------------------------------
    def _t_ul(self, el, d):  return self._list(el, d, ordered=False)
    def _t_ol(self, el, d):  return self._list(el, d, ordered=True)

    def _list(self, el: Tag, d: int, ordered: bool) -> str:
        start = int(el.get("start", 1)) if ordered else 0
        indent = "  " * d
        items: list[str] = []

        for i, li in enumerate(el.find_all("li", recursive=False)):
            text_parts: list[str] = []
            nested: list[str] = []
            for child in li.children:
                if isinstance(child, Tag) and child.name in ("ul", "ol"):
                    nested.append(self._el(child, d + 1))
                else:
                    text_parts.append(self._el(child, d))
            text = "".join(text_parts).strip().replace("\n\n", "\n")

            marker = f"{start + i}." if ordered else "-"
            item = f"{indent}{marker} {text}"
            for n in nested:
                item += "\n" + n.rstrip("\n")
            items.append(item)

        result = "\n".join(items)
        return result + "\n\n" if d == 0 else result + "\n"

    def _t_li(self, el, d):
        return self._inline(el)

    # --- Definition lists ---
    def _t_dl(self, el, d):
        cls = el.get("class", [])
        # Field lists (parameters, returns, etc.)
        if "field-list" in cls:
            return self._field_list(el, d)
        # Option lists
        if "option-list" in cls:
            return self._option_list(el, d)
        # API / object descriptions
        if any(c in cls for c in ("py", "c", "cpp", "describe",
                                   "function", "class", "method",
                                   "attribute", "data", "type")):
            return self._api_dl(el, d)
        # Generic dl
        parts: list[str] = []
        for c in el.children:
            if not isinstance(c, Tag):
                continue
            if c.name == "dt":
                parts.append(f"\n**{self._inline(c).strip()}**\n")
            elif c.name == "dd":
                body = self._children(c, d).strip()
                indented = "\n".join(
                    f"  {ln}" if ln.strip() else ""
                    for ln in body.split("\n")
                )
                parts.append(f"\n{indented}\n")
        return "".join(parts) + "\n"

    def _t_dt(self, el, d): return f"**{self._inline(el).strip()}**\n"
    def _t_dd(self, el, d): return f": {self._children(el, d).strip()}\n\n"

    # --- Field list (Parameters / Returns / Raises / …) ---
    def _field_list(self, el: Tag, d) -> str:
        parts: list[str] = []
        for c in el.children:
            if not isinstance(c, Tag):
                continue
            if c.name == "dt":
                parts.append(f"\n**{self._inline(c).strip()}**\n")
            elif c.name == "dd":
                body = self._children(c, d).strip()
                parts.append(f": {body}\n\n")
            # Sphinx sometimes uses <div class="field-body"> inside
            if c.name == "div" and "field-body" in c.get("class", []):
                body = self._children(c, d).strip()
                parts.append(f": {body}\n\n")
        return "".join(parts)

    # --- Option list ---
    def _option_list(self, el: Tag, d) -> str:
        parts: list[str] = []
        for c in el.children:
            if not isinstance(c, Tag):
                continue
            if c.name == "dt":
                parts.append(f"\n`{self._inline(c).strip()}`\n")
            elif c.name == "dd":
                body = self._children(c, d).strip()
                parts.append(f": {body}\n\n")
        return "".join(parts)

    # --- API description dl ---
    def _api_dl(self, el: Tag, d) -> str:
        parts: list[str] = []
        for c in el.children:
            if not isinstance(c, Tag):
                continue
            if c.name == "dt":
                sig = self._inline(c).strip()
                # Remove permalink
                sig = re.sub(r'\[\]\([^)]*"Permalink[^"]*"\)', "", sig)
                parts.append(f"\n```\n{sig}\n```\n\n")
            elif c.name == "dd":
                parts.append(self._children(c, d))
        return "".join(parts)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def _t_table(self, el: Tag, d) -> str:
        self._in_table = True
        headers, rows = self._table_data(el)
        self._in_table = False

        if not headers and not rows:
            return self._children(el, d)
        if not headers and rows:
            headers = rows.pop(0)
        ncols = max(len(headers), max((len(r) for r in rows), default=0))
        if ncols == 0:
            return ""

        def pad(row):
            return (row + [""] * ncols)[:ncols]

        headers = pad(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * ncols) + " |",
        ]
        for r in rows:
            cells = [c.replace("|", "\\|") for c in pad(r)]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n\n"

    def _table_data(self, tbl: Tag):
        headers: list[str] = []
        rows: list[list[str]] = []

        thead = tbl.find("thead")
        if thead:
            for tr in thead.find_all("tr", recursive=False):
                hrow = []
                for c in tr.find_all(["th", "td"], recursive=False):
                    hrow.append(self._inline(c).strip().replace("\n", " "))
                if hrow:
                    headers = hrow

        tbody = tbl.find("tbody") or tbl
        for tr in tbody.find_all("tr", recursive=False):
            if thead and tr.parent == thead:
                continue
            row: list[str] = []
            all_th = True
            for c in tr.find_all(["th", "td"], recursive=False):
                if c.name == "td":
                    all_th = False
                row.append(self._inline(c).strip().replace("\n", " "))
            if row:
                if all_th and not headers:
                    headers = row
                else:
                    rows.append(row)
        return headers, rows

    def _t_thead(self, e, d):    return ""
    def _t_tbody(self, e, d):    return ""
    def _t_tfoot(self, e, d):    return ""
    def _t_tr(self, e, d):       return ""
    def _t_th(self, e, d):       return ""
    def _t_td(self, e, d):       return ""
    def _t_colgroup(self, e, d): return ""
    def _t_col(self, e, d):      return ""
    def _t_caption(self, e, d):
        return f"*{self._inline(e).strip()}*\n\n"

    # ------------------------------------------------------------------
    # Inline elements
    # ------------------------------------------------------------------
    def _t_a(self, el: Tag, d) -> str:
        cls = el.get("class", [])
        if "headerlink" in cls:
            href = self._abs(el.get("href", ""))
            title = el.get("title", "Permalink to this headline")
            return f'[]({href} "{title}")'
        if "footnote-reference" in cls:
            return f'[^{el.get_text().strip()}]'
        if "fn-backref" in cls:
            return ""

        href = el.get("href", "")
        if not href:
            return self._inline(el)
        url = self._abs(href)
        text = self._inline(el).strip()
        if not text:
            return ""
        if text == url:
            return f"<{url}>"
        title = el.get("title", "")
        if title:
            return f'[{text}]({url} "{title}")'
        return f"[{text}]({url})"

    def _t_strong(self, el, d):
        t = self._inline(el).strip()
        return f"**{t}**" if t else ""
    _t_b = _t_strong

    def _t_em(self, el, d):
        t = self._inline(el).strip()
        return f"_{t}_" if t else ""
    _t_i = _t_em

    def _t_u(self, el, d):      return f"<u>{self._inline(el)}</u>"
    def _t_s(self, el, d):      return f"~~{self._inline(el)}~~"
    _t_strike = _t_s
    _t_del    = _t_s

    def _t_sup(self, el, d):    return f"<sup>{self._inline(el)}</sup>"
    def _t_sub(self, el, d):    return f"<sub>{self._inline(el)}</sub>"
    def _t_mark(self, el, d):   return f"=={self._inline(el)}=="
    def _t_cite(self, el, d):   return f"*{self._inline(el)}*"

    def _t_abbr(self, el, d):
        title = el.get("title", "")
        text = el.get_text()
        return f"{text} ({title})" if title else text

    def _t_span(self, el: Tag, d) -> str:
        cls = el.get("class", [])
        if "math" in cls or "MathJax" in cls:
            return f"\\({el.get_text().strip()}\\)"
        if "guilabel" in cls:
            return f"**{el.get_text()}**"
        if "menuselection" in cls:
            return f"*{el.get_text()}*"
        if "pre" in cls or "literal" in cls:
            return f"`{el.get_text()}`"
        if any(c in cls for c in ("command", "program", "file",
                                   "regexp", "makevar", "option")):
            return f"`{el.get_text()}`"
        # Highlighted search terms, etc.
        if "highlighted" in cls:
            return f"**{el.get_text()}**"
        return self._inline(el)

    # ------------------------------------------------------------------
    # Images & figures
    # ------------------------------------------------------------------
    def _t_img(self, el: Tag, d) -> str:
        src = el.get("src", "") or el.get("data-src", "")
        if not src:
            return ""
        url = self._abs(src)
        self._img_counter += 1
        alt = el.get("alt", "") or f"image_{self._img_counter}"
        return f"![{alt}]({url})"

    def _t_figure(self, el, d):
        return self._figure(el, d)

    def _t_figcaption(self, el, d):
        return f"\n{self._inline(el).strip()}\n\n"

    def _figure(self, el: Tag, d) -> str:
        parts: list[str] = []
        img = el.find("img")
        if img:
            parts.append(self._t_img(img, d))
        cap = el.find("figcaption") or el.find(class_="caption")
        if cap:
            txt = self._inline(cap).strip()
            txt = re.sub(r'\[\]\([^)]*"Permalink[^"]*"\)', "", txt)
            parts.append(f"\n\n{txt}")
        return "\n".join(parts) + "\n\n"

    # ------------------------------------------------------------------
    # Admonitions (note / warning / tip …)
    # ------------------------------------------------------------------
    def _admonition(self, el: Tag, d) -> str:
        title_el = (el.find(class_="admonition-title")
                    or el.find("p", class_="topic-title")
                    or el.find("p", class_="sidebar-title"))
        cls = el.get("class", [])
        kind = "Note"
        for c in cls:
            if c in ("note", "warning", "tip", "important", "caution",
                      "danger", "error", "hint", "seealso", "todo",
                      "admonition"):
                kind = c.capitalize()
                break
        if title_el:
            kind = title_el.get_text().strip()

        body_parts: list[str] = []
        skip_cls = {"admonition-title", "topic-title", "sidebar-title"}
        for child in el.children:
            if isinstance(child, Tag):
                cc = set(child.get("class", []))
                if cc & skip_cls:
                    continue
                body_parts.append(self._el(child, d))
        body = "".join(body_parts).strip()

        lines = [f"> **{kind}**", ">"]
        for ln in body.split("\n"):
            lines.append(f"> {ln}")
        return "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Version directives
    # ------------------------------------------------------------------
    def _version_directive(self, el: Tag, d) -> str:
        cls = el.get("class", [])
        kind = "Note"
        for c in cls:
            if c == "versionadded":
                kind = "Added"
            elif c == "versionchanged":
                kind = "Changed"
            elif c == "deprecated":
                kind = "Deprecated"
            elif c == "versionremoved":
                kind = "Removed"

        body = self._children(el, d).strip()
        return f"\n> **{kind}:** {body}\n\n"

    # ------------------------------------------------------------------
    # Sphinx tabs
    # ------------------------------------------------------------------
    def _sphinx_tabs(self, el: Tag, d) -> str:
        parts: list[str] = []
        # Collect tab labels
        tabs = el.find_all("div", class_="sphinx-tabs-tab")
        panels = el.find_all("div", class_="sphinx-tabs-panel")

        for tab_label, panel in zip(tabs, panels):
            label = tab_label.get_text().strip()
            body = self._children(panel, d).strip()
            parts.append(f"\n**{label}**\n\n{body}\n")

        # If no structured tabs found, just render children
        if not parts:
            return self._children(el, d)
        return "\n".join(parts) + "\n"

    def _sd_tab_set(self, el: Tag, d) -> str:
        parts: list[str] = []
        labels = el.find_all("label", class_="sd-tab-label")
        panels = el.find_all("div", class_="sd-tab-content")

        for label, panel in zip(labels, panels):
            title = label.get_text().strip()
            body = self._children(panel, d).strip()
            parts.append(f"\n**{title}**\n\n{body}\n")

        if not parts:
            return self._children(el, d)
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    # Line block
    # ------------------------------------------------------------------
    def _line_block(self, el: Tag, d) -> str:
        lines: list[str] = []
        for child in el.children:
            if isinstance(child, Tag):
                if "line-block" in child.get("class", []):
                    nested = self._line_block(child, d)
                    for ln in nested.strip().split("\n"):
                        lines.append(f"  {ln}")
                elif child.name == "div" and "line" in child.get("class", []):
                    txt = self._inline(child).strip()
                    lines.append(txt if txt else "")
                else:
                    lines.append(self._inline(child).strip())
        return "\n".join(f"| {ln}" for ln in lines) + "\n\n"

    # ------------------------------------------------------------------
    # Details / Summary
    # ------------------------------------------------------------------
    def _t_details(self, el, d):
        summary = el.find("summary")
        stxt = self._inline(summary).strip() if summary else "Details"
        parts = [self._el(c, d) for c in el.children
                 if isinstance(c, Tag) and c.name != "summary"]
        body = "".join(parts).strip()
        return (f"<details>\n<summary>{stxt}</summary>\n\n"
                f"{body}\n\n</details>\n\n")

    def _t_summary(self, el, d): return ""

    # ------------------------------------------------------------------
    # Footnotes
    # ------------------------------------------------------------------
    def _t_footnote(self, el: Tag, d) -> str:
        # Sphinx <aside class="footnote"> or <div class="footnote">
        label_el = el.find("span", class_="label") or el.find("span", class_="fn-bracket")
        label = ""
        if label_el:
            label = label_el.get_text().strip().strip("[]")

        body_parts = []
        for child in el.children:
            if isinstance(child, Tag):
                if child == label_el or child.find_parent() == label_el:
                    continue
                if "label" in child.get("class", []):
                    continue
                if "backrefs" in child.get("class", []):
                    continue
                body_parts.append(self._el(child, d))
        body = "".join(body_parts).strip()

        if label:
            return f"[^{label}]: {body}\n\n"
        return f"> {body}\n\n"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _abs(self, href: str) -> str:
        if not href:
            return ""
        if href.startswith(("http://", "https://", "mailto://", "//")):
            return href
        return urllib.parse.urljoin(self.base_url + "/", href)

    @staticmethod
    def _cleanup(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(ln.rstrip() for ln in text.split("\n"))
        return text.strip() + "\n"


# =============================================================================
# Crawler
# =============================================================================
class NvidiaDocCrawler:
    """
    Generic crawler for Sphinx-based NVIDIA documentation.

    Orchestrates:  URL resolution → TOC parsing → page fetching →
    HTML pre-processing → section splitting → Markdown conversion →
    file writing.
    """

    def __init__(self, *,
                 url: str,
                 output_dir: Optional[str] = None,
                 delay: float              = DEFAULT_DELAY,
                 split_depth: int          = 2,
                 single_file: bool         = False,
                 download_images: bool     = False):
        self.input_url       = url
        self.delay           = delay
        self.split_depth     = split_depth
        self.single_file     = single_file
        self.download_images = download_images

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"),
            "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language":  "en-US,en;q=0.9",
            "Accept-Encoding":  "gzip, deflate, br",
        })

        # Resolve URLs
        self.base_url, self.toc_url, self.slug = \
            DocURLResolver.resolve(url)

        # Output directory: user-supplied or auto from slug
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"{self.slug}_markdown")

        self.conv: Optional[MarkdownConverter] = None   # created after URL resolution
        self.sections: OrderedDict[str, SectionInfo] = OrderedDict()
        self._pages_cache: Dict[str, BeautifulSoup] = {}

        # Determined later
        self._doc_title: str = ""

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _fetch(self, url: str) -> Optional[str]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log.info("  GET %s  (attempt %d)", url, attempt)
                r = self.session.get(url, timeout=60)
                r.raise_for_status()
                r.encoding = r.apparent_encoding or "utf-8"
                time.sleep(self.delay)
                return r.text
            except requests.exceptions.HTTPError as exc:
                code = exc.response.status_code if exc.response else 0
                if code == 429:
                    w = RETRY_WAIT * attempt * 2
                    log.warning("  429 – waiting %.0fs …", w)
                    time.sleep(w)
                elif code >= 500:
                    time.sleep(RETRY_WAIT * attempt)
                else:
                    log.error("  HTTP %d – skipping", code)
                    return None
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                log.warning("  %s – retrying in %.0fs …",
                            type(exc).__name__, RETRY_WAIT * attempt)
                time.sleep(RETRY_WAIT * attempt)
            except requests.exceptions.RequestException as exc:
                log.error("  %s", exc)
                return None
        log.error("  FAILED after %d attempts: %s", MAX_RETRIES, url)
        return None

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        if url in self._pages_cache:
            return self._pages_cache[url]
        html = self._fetch(url)
        if html is None:
            return None
        soup = BeautifulSoup(html, HTML_PARSER)
        HTMLPreprocessor.preprocess(soup)
        self._pages_cache[url] = soup
        return soup

    # ------------------------------------------------------------------
    # Document title detection
    # ------------------------------------------------------------------
    def _detect_title(self, soup: BeautifulSoup) -> str:
        """Try to extract the document title from the page."""
        # <title> tag
        title_tag = soup.find("title")
        if title_tag:
            raw = title_tag.get_text().strip()
            # Remove trailing " — xxx documentation" etc.
            raw = re.sub(r"\s*[—–-]\s*.*documentation.*$", "", raw, flags=re.I)
            raw = re.sub(r"\s*[—–-]\s*NVIDIA.*$", "", raw, flags=re.I)
            if raw:
                return raw.strip()
        h1 = soup.find("h1")
        if h1:
            return self.conv._heading_text(h1) if self.conv else h1.get_text().strip()
        return self.slug.replace("-", " ").title()

    # ------------------------------------------------------------------
    # TOC parsing
    # ------------------------------------------------------------------
    def _parse_toc(self, soup: BeautifulSoup) -> List[Dict]:
        wrap = (soup.find("div", class_="toctree-wrapper")
                or soup.find("div", class_="contents")
                or soup.find("nav", class_="contents")
                or soup.find("div", role="main")
                or soup)

        def walk(ul, depth=0):
            if not ul:
                return []
            out = []
            for li in ul.find_all("li", recursive=False):
                a = li.find("a", recursive=False) or li.find("a")
                if not a:
                    continue
                href  = a.get("href", "")
                title = a.get_text().strip()
                anchor = href.split("#")[-1] if "#" in href else ""
                entry = dict(title=title, href=href, anchor=anchor,
                             depth=depth, children=[])
                sub = li.find("ul", recursive=False)
                if sub:
                    entry["children"] = walk(sub, depth + 1)
                out.append(entry)
            return out

        ul = wrap.find("ul")
        return walk(ul) if ul else []

    @staticmethod
    def _flatten_toc(entries, depth=0) -> Generator:
        for e in entries:
            yield {**e, "depth": depth}
            yield from NvidiaDocCrawler._flatten_toc(
                e.get("children", []), depth + 1)

    def _toc_is_multipage(self, entries) -> bool:
        pages = set()
        for e in self._flatten_toc(entries):
            href = e["href"]
            page = href.split("#")[0].strip()
            if page:
                pages.add(page)
        return len(pages) > 1

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------
    def _content_root(self, soup: BeautifulSoup) -> Tag:
        for sel in (
            {"name": "div",     "attrs": {"class": "body"}},
            {"name": "div",     "attrs": {"role": "main"}},
            {"name": "div",     "attrs": {"class": "document"}},
            {"name": "main"},
            {"name": "article"},
        ):
            el = soup.find(**sel)
            if el:
                return el
        return soup.find("body") or soup

    def _extract_sections(self, soup: BeautifulSoup) -> OrderedDict:
        root = self._content_root(soup)
        sections: OrderedDict[str, SectionInfo] = OrderedDict()

        candidates = root.find_all("section", recursive=True)
        if not candidates:
            candidates = root.find_all(
                "div", class_="section", recursive=True)

        for sec in candidates:
            sid = sec.get("id", "")
            if not sid:
                continue

            heading = sec.find(
                ["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False)
            if not heading:
                heading = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"])

            if heading:
                htxt = self.conv._heading_text(heading)
                hlvl = int(heading.name[1])
            else:
                htxt = sid.replace("-", " ").title()
                hlvl = 1

            num_m = re.match(r"^([\d.]+\.?)\s*", htxt)
            snum  = num_m.group(1).rstrip(".") if num_m else ""

            depth = 0
            parent_id = None
            p = sec.parent
            while p:
                if isinstance(p, Tag) and (
                    p.name == "section"
                    or ("section" in p.get("class", []))
                ):
                    depth += 1
                    if p.get("id") and parent_id is None:
                        parent_id = p["id"]
                p = getattr(p, "parent", None)

            sections[sid] = SectionInfo(
                id=sid, title=htxt, section_number=snum,
                heading_level=hlvl, element=sec,
                parent_id=parent_id, depth=depth,
            )

        for sid, si in sections.items():
            if si.parent_id and si.parent_id in sections:
                sections[si.parent_id].children_ids.append(sid)

        return sections

    # ------------------------------------------------------------------
    # Multi-page support
    # ------------------------------------------------------------------
    def _crawl_multipage(self, toc_entries):
        seen_pages: set = set()
        all_sections: OrderedDict[str, SectionInfo] = OrderedDict()

        for entry in self._flatten_toc(toc_entries):
            page = entry["href"].split("#")[0].strip()
            if not page or page in seen_pages:
                continue
            seen_pages.add(page)
            url = urllib.parse.urljoin(self.base_url, page)
            soup = self._get_soup(url)
            if soup is None:
                continue
            # Update converter base_url for this page so relative links resolve
            page_base = url.rsplit("/", 1)[0] + "/" if "/" in url else url
            self.conv.base_url = page_base.rstrip("/")
            secs = self._extract_sections(soup)
            all_sections.update(secs)

        # Restore base_url
        self.conv.base_url = self.base_url.rstrip("/")
        return all_sections

    # ------------------------------------------------------------------
    # Image downloading
    # ------------------------------------------------------------------
    def _download_images_from_md(self, md_text: str, section_id: str) -> str:
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)

        counter = [0]

        def replacer(m):
            alt = m.group(1)
            url = m.group(2)
            if url.startswith(("data:", "images/")):
                return m.group(0)
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
            except Exception as exc:
                log.warning("    Could not download image %s: %s", url, exc)
                return m.group(0)

            ct = resp.headers.get("Content-Type", "")
            ext = ".png"
            if "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            elif "gif" in ct:
                ext = ".gif"
            elif "svg" in ct:
                ext = ".svg"
            elif "webp" in ct:
                ext = ".webp"

            counter[0] += 1
            fname = re.sub(r"[^\w\-]", "_", section_id)
            fname = f"{fname}_{counter[0]}{ext}"
            (img_dir / fname).write_bytes(resp.content)
            local = f"images/{fname}"
            return f"![{alt}]({local})"

        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replacer, md_text)

    # ------------------------------------------------------------------
    # File naming
    # ------------------------------------------------------------------
    @staticmethod
    def _filename(si: SectionInfo, index: int) -> str:
        if si.section_number:
            parts = si.section_number.split(".")
            prefix = "-".join(p.zfill(2) for p in parts)
        else:
            prefix = f"{index:04d}"
        slug = re.sub(r"[^\w\-]", "-", si.id)
        slug = re.sub(r"-{2,}", "-", slug).strip("-")[:80]
        return f"{prefix}_{slug}.md"

    # ------------------------------------------------------------------
    # YAML front-matter
    # ------------------------------------------------------------------
    def _front_matter(self, title: str, section_num: str, url: str) -> str:
        safe_title = title.replace('"', '\\"')
        return (
            f"---\n"
            f'title: "{safe_title}"\n'
            f'section: "{section_num}"\n'
            f'source: "{url}"\n'
            f"---\n\n"
        )

    # ------------------------------------------------------------------
    # Index file
    # ------------------------------------------------------------------
    def _save_index(self, sections_to_save: List[SectionInfo]):
        doc_title = self._doc_title or self.slug.replace("-", " ").title()
        lines = [
            "---",
            f"title: {doc_title} — Index",
            f"source: {self.base_url}",
            f"crawled: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "---", "",
            f"# {doc_title}", "",
            f"Source: <{self.base_url}>", "",
            "## Table of Contents", "",
        ]
        for i, si in enumerate(sections_to_save):
            indent = "  " * si.depth
            fn = self._filename(si, i)
            label = f"{si.section_number}. " if si.section_number else ""
            clean = re.sub(r"^[\d.]+\s*", "", si.title)
            lines.append(f"{indent}- [{label}{clean}](./{fn})")
        lines.append("")
        f = self.output_dir / "00_index.md"
        f.write_text("\n".join(lines), encoding="utf-8")
        log.info("  Saved index → %s", f.name)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> bool:
        log.info("=" * 60)
        log.info("NVIDIA Documentation Crawler & Markdown Converter")
        log.info("=" * 60)
        log.info("Input URL  : %s", self.input_url)
        log.info("Base URL   : %s", self.base_url)
        log.info("TOC URL    : %s", self.toc_url)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output dir : %s", self.output_dir.resolve())

        # Initialise converter with resolved base URL
        self.conv = MarkdownConverter(self.base_url)

        # --- 1. Verify & fetch TOC ----------------------------------------
        log.info("\n[1/4] Fetching table of contents …")
        verified_toc = DocURLResolver.verify_toc(
            self.toc_url, self.base_url, self.session)
        if verified_toc != self.toc_url:
            log.info("  Adjusted TOC URL → %s", verified_toc)
            self.toc_url = verified_toc

        toc_soup = self._get_soup(self.toc_url)
        if toc_soup:
            self._doc_title = self._detect_title(toc_soup)
        toc_entries = self._parse_toc(toc_soup) if toc_soup else []
        n_toc = sum(1 for _ in self._flatten_toc(toc_entries))
        log.info("  %d TOC entries found", n_toc)
        log.info("  Document title: %s", self._doc_title)

        # --- 2. Detect single-page vs multi-page -------------------------
        multipage = self._toc_is_multipage(toc_entries) if toc_entries else False

        if multipage:
            log.info("\n[2/4] Multi-page doc detected — crawling pages …")
            self.sections = self._crawl_multipage(toc_entries)
        else:
            log.info("\n[2/4] Single-page doc — fetching main page …")
            # For single-page, the main content is on the base URL (index.html)
            main_url = self.base_url
            soup = self._get_soup(main_url)
            if not soup:
                # Try index.html explicitly
                soup = self._get_soup(self.base_url + "index.html")
            if not soup:
                log.error("  FATAL: cannot fetch main page.")
                return False
            if not self._doc_title:
                self._doc_title = self._detect_title(soup)

            sz = len(str(soup)) / 1024 / 1024
            log.info("  Parsed %.1f MB of HTML", sz)

            log.info("\n[3/4] Extracting sections …")
            self.sections = self._extract_sections(soup)

        log.info("  %d sections extracted", len(self.sections))
        if not self.sections:
            log.error("  No sections found — aborting.")
            return False

        # --- 3/4. Convert & save -----------------------------------------
        if self.single_file:
            log.info("\n[4/4] Writing single Markdown file …")
            self._write_single()
        else:
            log.info("\n[4/4] Writing individual Markdown files "
                     "(split-depth=%d) …", self.split_depth)
            self._write_split()

        log.info("\n" + "=" * 60)
        log.info("Done!  Files in: %s", self.output_dir.resolve())
        log.info("=" * 60)
        return True

    # ------------------------------------------------------------------
    def _write_single(self):
        parts: list[str] = []
        for si in self.sections.values():
            parts.append(self.conv.convert_section_body(
                si.element, include_children=False))
        md = "\n".join(parts)
        slug = re.sub(r"[^\w\-]", "-", self.slug).strip("-")
        header = (
            "---\n"
            f"title: {self._doc_title} (complete)\n"
            f"source: {self.base_url}\n"
            f"crawled: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "---\n\n"
        )
        out = self.output_dir / f"{slug}-complete.md"
        out.write_text(header + md, encoding="utf-8")
        log.info("  → %s  (%.1f KB)", out.name, out.stat().st_size / 1024)

    # ------------------------------------------------------------------
    def _write_split(self):
        to_save: list[SectionInfo] = []
        for si in self.sections.values():
            if self.split_depth == 0 or si.depth <= self.split_depth:
                to_save.append(si)
        if not to_save:
            to_save = list(self.sections.values())

        self._save_index(to_save)

        total = len(to_save)
        for idx, si in enumerate(to_save, 1):
            children_separate = any(
                cid in {s.id for s in to_save} for cid in si.children_ids)
            md = self.conv.convert_section_body(
                si.element, include_children=not children_separate)
            if not md.strip():
                continue

            if self.download_images:
                md = self._download_images_from_md(md, si.id)

            fm = self._front_matter(si.title, si.section_number,
                                    f"{self.base_url}#{si.id}")
            fn = self._filename(si, idx)
            out = self.output_dir / fn
            out.write_text(fm + md, encoding="utf-8")
            kb = out.stat().st_size / 1024
            log.info("  [%d/%d] %s  (%.1f KB)", idx, total, fn, kb)

        log.info("  Total: %d files + index", total)


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Crawl any Sphinx-based NVIDIA documentation site → Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html
  %(prog)s https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html
  %(prog)s https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ -o best_practices
  %(prog)s <URL> --single-file
  %(prog)s <URL> --split-depth 3
  %(prog)s <URL> --download-images
  %(prog)s <URL> -d 0.3 -v
        """,
    )
    ap.add_argument("url",
                    help="URL of the documentation (TOC page, index, or base dir)")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="output directory  (default: auto from doc name)")
    ap.add_argument("-d", "--delay", type=float, default=DEFAULT_DELAY,
                    help=f"seconds between requests  (default: {DEFAULT_DELAY})")
    ap.add_argument("--split-depth", type=int, default=2,
                    help="section nesting depth for file splitting (0=all, default=2)")
    ap.add_argument("--single-file", action="store_true",
                    help="save everything as one Markdown file")
    ap.add_argument("--download-images", action="store_true",
                    help="download images to local images/ folder")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="debug-level logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ok = NvidiaDocCrawler(
        url=args.url,
        output_dir=args.output_dir,
        delay=args.delay,
        split_depth=args.split_depth,
        single_file=args.single_file,
        download_images=args.download_images,
    ).run()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
