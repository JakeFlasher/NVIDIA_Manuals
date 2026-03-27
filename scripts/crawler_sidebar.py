#!/usr/bin/env python3
"""
NVIDIA Multi-Page Documentation Crawler & Markdown Converter
=============================================================
Crawler for Sphinx-based NVIDIA documentation that spans multiple HTML pages.

Handles single-page AND multi-page documentation sites, including:
  - CUDA Programming Guide:  .../cuda-programming-guide/index.html
  - cuTile Python:            .../cutile-python/index.html
  - Tile IR:                  .../tile-ir/latest/sections/introduction.html
  - PTX ISA:                  .../parallel-thread-execution/
  - Any other Sphinx-based NVIDIA doc site

Key improvements over the single-page crawler:
  - Automatic discovery of all pages from sidebar nav, toctree, rel-next chain
  - BFS-based recursive page discovery as robust fallback
  - Cross-page section ordering with duplicate-ID handling
  - Per-page base-URL for correct relative-link resolution
  - Fallback "whole-page" section when no <section> tags exist
  - --list-pages mode for debugging discovery without crawling

Requirements:
    pip install requests beautifulsoup4 lxml

Usage:
    python nvidia_multipage_crawler.py <URL>
    python nvidia_multipage_crawler.py <URL> -o my_docs
    python nvidia_multipage_crawler.py <URL> --single-file
    python nvidia_multipage_crawler.py <URL> --list-pages
    python nvidia_multipage_crawler.py <URL> --max-pages 50

Examples:
    python nvidia_multipage_crawler.py https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
    python nvidia_multipage_crawler.py https://docs.nvidia.com/cuda/cutile-python/index.html
    python nvidia_multipage_crawler.py https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html
"""

import os
import re
import sys
import time
import logging
import argparse
import urllib.parse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass, field
from collections import OrderedDict, deque

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
DEFAULT_DELAY     = 0.8
MAX_RETRIES       = 3
RETRY_WAIT        = 5.0
DEFAULT_MAX_PAGES = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nvidia_multipage_crawler")


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class PageEntry:
    """A discovered documentation page."""
    url: str
    title: str
    depth: int = 0
    order: int = 0


@dataclass
class SectionInfo:
    """Metadata about one <section> extracted from HTML."""
    id: str
    title: str
    section_number: str
    heading_level: int
    element: Tag
    page_url: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    global_order: int = 0


# =============================================================================
# HTML Pre-processor
# =============================================================================
class HTMLPreprocessor:
    """
    In-place mutations on a BeautifulSoup tree to ensure *all* content is
    visible — hidden tabs, collapsed details, display:none blocks, etc.
    """

    @staticmethod
    def preprocess(soup: BeautifulSoup) -> BeautifulSoup:
        for sel in (
            {"class_": lambda c: c and any("copy" in x.lower() for x in
                                            (c if isinstance(c, list) else [c]))},
            {"attrs": {"data-clipboard-target": True}},
            {"class_": "copybtn"},
            {"class_": "copybutton"},
        ):
            for btn in soup.find_all("button", **sel):
                btn.decompose()

        for table in soup.find_all("table", class_="highlighttable"):
            code_cell = table.find("td", class_="code")
            if code_cell:
                hl = code_cell.find(
                    "div", class_=lambda c: c and "highlight" in
                    (" ".join(c) if isinstance(c, list) else c))
                if hl:
                    table.replace_with(hl)
        for div in soup.find_all("div", class_="linenodiv"):
            div.decompose()
        for td in soup.find_all("td", class_="linenos"):
            td.decompose()

        for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            el["style"] = re.sub(
                r"display\s*:\s*none\s*;?\s*", "", el.get("style", ""),
                flags=re.I,
            )

        for el in soup.find_all(attrs={"hidden": True}):
            del el["hidden"]

        for panel in soup.find_all(
            "div", class_=lambda c: c and (
                "sphinx-tabs-panel" in (c if isinstance(c, list) else [c])
            )
        ):
            panel.attrs.pop("style", None)
            panel["aria-hidden"] = "false"
        for panel in soup.find_all("div", class_="sd-tab-content"):
            panel.attrs.pop("style", None)
            cls = panel.get("class", [])
            panel["class"] = [c for c in cls if c != "d-none"]

        for el in soup.find_all(class_="collapse"):
            cls = el.get("class", [])
            el["class"] = [c for c in cls if c not in ("collapse", "collapsed")]

        for el in soup.find_all(attrs={"aria-hidden": "true"}):
            tag = el.name or ""
            if tag in ("div", "section", "article", "pre", "code",
                       "table", "ul", "ol", "dl", "p", "span"):
                el["aria-hidden"] = "false"

        return soup


# =============================================================================
# HTML -> Markdown Converter
# =============================================================================
class MarkdownConverter:
    """
    Comprehensive HTML-to-Markdown converter for Sphinx / NVIDIA docs.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._img_counter = 0
        self._in_pre   = False
        self._in_table = False

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

    def _el(self, node, depth: int = 0) -> str:
        if node is None:
            return ""
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return str(node)
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

    # --- headings ---------------------------------------------------------
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
        if p and isinstance(p, Tag) and p.name in ("section", "div") and p.get("id"):
            return p["id"]
        hl = el.find("a", class_="headerlink")
        if hl and hl.get("href", "").startswith("#"):
            return hl["href"][1:]
        return ""

    # --- block elements ---------------------------------------------------
    def _t_p(self, el, d):
        cls = el.get("class", [])
        if "rubric" in cls:
            t = self._inline(el).strip()
            return f"\n**{t}**\n\n" if t else ""
        t = self._inline(el).strip()
        return f"{t}\n\n" if t else ""

    def _t_div(self, el, d):
        cls = el.get("class", [])
        if any(c in cls for c in ("admonition", "note", "warning", "tip",
                                   "important", "caution", "danger",
                                   "error", "hint", "seealso", "todo")):
            return self._admonition(el, d)
        if any(c.startswith("highlight") for c in cls):
            pre = el.find("pre")
            return self._t_pre(pre, d) if pre else self._children(el, d)
        if "literal-block-wrapper" in cls:
            return self._literal_block_wrapper(el, d)
        if any(c in cls for c in ("figure", "align-default",
                                   "align-center", "align-left", "align-right")):
            return self._figure(el, d)
        if "math" in cls:
            return f"\n$$\n{el.get_text().strip()}\n$$\n\n"
        if any(c in cls for c in ("versionadded", "versionchanged",
                                   "deprecated", "versionremoved")):
            return self._version_directive(el, d)
        if "sphinx-tabs" in cls:
            return self._sphinx_tabs(el, d)
        if "sd-tab-set" in cls:
            return self._sd_tab_set(el, d)
        if any(c in cls for c in ("topic", "sidebar", "contents")):
            return self._admonition(el, d)
        if "container" in cls:
            return self._children(el, d)
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

    # --- code -------------------------------------------------------------
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

    # --- lists ------------------------------------------------------------
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

    def _t_li(self, el, d):  return self._inline(el)

    def _t_dl(self, el, d):
        cls = el.get("class", [])
        if "field-list" in cls:
            return self._field_list(el, d)
        if "option-list" in cls:
            return self._option_list(el, d)
        if any(c in cls for c in ("py", "c", "cpp", "describe",
                                   "function", "class", "method",
                                   "attribute", "data", "type")):
            return self._api_dl(el, d)
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
            if c.name == "div" and "field-body" in c.get("class", []):
                body = self._children(c, d).strip()
                parts.append(f": {body}\n\n")
        return "".join(parts)

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

    def _api_dl(self, el: Tag, d) -> str:
        parts: list[str] = []
        for c in el.children:
            if not isinstance(c, Tag):
                continue
            if c.name == "dt":
                sig = self._inline(c).strip()
                sig = re.sub(r'\[\]\([^)]*"Permalink[^"]*"\)', "", sig)
                parts.append(f"\n```\n{sig}\n```\n\n")
            elif c.name == "dd":
                parts.append(self._children(c, d))
        return "".join(parts)

    # --- tables -----------------------------------------------------------
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
    def _t_caption(self, e, d):  return f"*{self._inline(e).strip()}*\n\n"

    # --- inline -----------------------------------------------------------
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
        if "highlighted" in cls:
            return f"**{el.get_text()}**"
        return self._inline(el)

    # --- images & figures -------------------------------------------------
    def _t_img(self, el: Tag, d) -> str:
        src = el.get("src", "") or el.get("data-src", "")
        if not src:
            return ""
        url = self._abs(src)
        self._img_counter += 1
        alt = el.get("alt", "") or f"image_{self._img_counter}"
        return f"![{alt}]({url})"

    def _t_figure(self, el, d):   return self._figure(el, d)
    def _t_figcaption(self, el, d): return f"\n{self._inline(el).strip()}\n\n"

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

    # --- admonitions ------------------------------------------------------
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

    # --- version directives -----------------------------------------------
    def _version_directive(self, el: Tag, d) -> str:
        cls = el.get("class", [])
        kind = "Note"
        for c in cls:
            if c == "versionadded":    kind = "Added"
            elif c == "versionchanged": kind = "Changed"
            elif c == "deprecated":     kind = "Deprecated"
            elif c == "versionremoved": kind = "Removed"
        body = self._children(el, d).strip()
        return f"\n> **{kind}:** {body}\n\n"

    # --- sphinx tabs ------------------------------------------------------
    def _sphinx_tabs(self, el: Tag, d) -> str:
        parts: list[str] = []
        tabs = el.find_all("div", class_="sphinx-tabs-tab")
        panels = el.find_all("div", class_="sphinx-tabs-panel")
        for tab_label, panel in zip(tabs, panels):
            label = tab_label.get_text().strip()
            body = self._children(panel, d).strip()
            parts.append(f"\n**{label}**\n\n{body}\n")
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

    # --- line block -------------------------------------------------------
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

    # --- details / summary ------------------------------------------------
    def _t_details(self, el, d):
        summary = el.find("summary")
        stxt = self._inline(summary).strip() if summary else "Details"
        parts = [self._el(c, d) for c in el.children
                 if isinstance(c, Tag) and c.name != "summary"]
        body = "".join(parts).strip()
        return (f"<details>\n<summary>{stxt}</summary>\n\n"
                f"{body}\n\n</details>\n\n")

    def _t_summary(self, el, d): return ""

    # --- footnotes --------------------------------------------------------
    def _t_footnote(self, el: Tag, d) -> str:
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

    # --- helpers ----------------------------------------------------------
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
# Navigation Discovery  (IMPROVED — tries all selectors, picks best result)
# =============================================================================
class NavigationDiscovery:
    """
    Discovers all pages in a multi-page documentation set.

    Strategies (tried in order, best result wins):
      1. Sidebar navigation (Furo / PyData / RTD / NVIDIA themes)
         — tries ALL matching selectors and picks the one with the most links
      2. Toctree wrapper in main content area
      3. Internal .html links in content area
    The crawler itself adds additional fallbacks:
      4. Following rel="next" / class="next-page" chains
      5. BFS-based recursive page discovery
    """

    def __init__(self, doc_root_url: str, base_url: str = ""):
        self.doc_root_url = doc_root_url.rstrip("/") + "/"
        # IMPROVED: also track base_url for more flexible internal checks
        self.base_url = (base_url.rstrip("/") + "/") if base_url else self.doc_root_url

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def discover(self, soup: BeautifulSoup, page_url: str) -> List[PageEntry]:
        """Return an ordered list of page URLs from the best available source."""

        pages = self._from_sidebar(soup, page_url)
        if pages and len(pages) > 1:
            log.info("    -> %d pages via sidebar navigation", len(pages))
            return pages

        pages = self._from_toctree(soup, page_url)
        if pages and len(pages) > 1:
            log.info("    -> %d pages via toctree wrapper", len(pages))
            return pages

        pages = self._from_internal_links(soup, page_url)
        if pages:
            log.info("    -> %d pages via internal-link scan", len(pages))
            return pages

        log.info("    -> no extra pages found on this page")
        return [PageEntry(url=page_url, title="", depth=0, order=0)]

    # ------------------------------------------------------------------
    # Strategy 1 — sidebar navigation  (IMPROVED)
    # ------------------------------------------------------------------
    _SIDEBAR_SELECTORS = [
        # Furo theme
        ("div",   {"class_": "sidebar-tree"}),
        ("div",   {"class_": "sidebar-scroll"}),
        # PyData Sphinx theme
        ("nav",   {"id": "bd-docs-nav"}),
        ("div",   {"class_": "bd-toc-item"}),
        # Read the Docs theme
        ("div",   {"class_": "wy-menu-vertical"}),
        ("div",   {"class_": "wy-menu"}),
        ("div",   {"class_": "sphinxsidebarwrapper"}),
        # NVIDIA custom themes
        ("nav",   {"class_": "wy-nav-side"}),
        ("nav",   {"attrs": {"aria-label": "main navigation"}}),
        ("nav",   {"attrs": {"aria-label": "Navigation menu"}}),
        ("aside", {"class_": "sidebar-drawer"}),
        ("div",   {"class_": "sidebar-container"}),
        # Sphinx Book Theme
        ("nav",   {"class_": "bd-links"}),
        # Generic sidebar / toc containers
        ("nav",   {"class_": "sidebar"}),
        ("div",   {"class_": "sidebar"}),
        ("div",   {"class_": "toc"}),
        ("nav",   {"class_": "toc"}),
        ("div",   {"class_": "toctree-wrapper"}),
    ]

    def _from_sidebar(self, soup: BeautifulSoup,
                      page_url: str) -> List[PageEntry]:
        # IMPROVED: try ALL selectors and pick the one that yields the most
        # page links, instead of stopping at the first match.
        best_pages: List[PageEntry] = []
        best_label = ""
        for tag, attrs in self._SIDEBAR_SELECTORS:
            try:
                containers = soup.find_all(tag, **attrs)
            except Exception:
                continue
            for nav_el in containers:
                pages = self._extract_page_links(nav_el, page_url)
                if len(pages) > len(best_pages):
                    best_pages = pages
                    best_label = f"<{tag} {attrs}>"
                    log.debug("    sidebar candidate %s -> %d pages",
                              best_label, len(pages))
        if best_pages:
            log.debug("    best sidebar: %s (%d pages)",
                      best_label, len(best_pages))
        return best_pages

    # ------------------------------------------------------------------
    # Strategy 2 — toctree-wrapper in content area
    # ------------------------------------------------------------------
    def _from_toctree(self, soup: BeautifulSoup,
                      page_url: str) -> List[PageEntry]:
        content = self._content_area(soup)
        if not content:
            return []
        wrappers = content.find_all("div", class_="toctree-wrapper")
        if not wrappers:
            ctn = content.find("div", class_="contents")
            if ctn:
                wrappers = [ctn]
        pages: List[PageEntry] = []
        for w in wrappers:
            pages.extend(self._extract_page_links(w, page_url))
        return self._dedupe(pages)

    # ------------------------------------------------------------------
    # Strategy 3 — all internal .html links in content
    # ------------------------------------------------------------------
    def _from_internal_links(self, soup: BeautifulSoup,
                             page_url: str) -> List[PageEntry]:
        content = self._content_area(soup) or soup
        pages: List[PageEntry] = []
        seen: Set[str] = set()
        order = 0
        for a in content.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("#", "mailto:", "javascript:")):
                continue
            bare = href.split("#")[0].strip()
            if not bare or not bare.endswith(".html"):
                continue
            full = urllib.parse.urljoin(page_url, bare)
            if not self._is_internal(full) or full in seen:
                continue
            seen.add(full)
            pages.append(PageEntry(url=full, title=a.get_text().strip(),
                                   depth=0, order=order))
            order += 1
        return pages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_page_links(self, container: Tag,
                            page_url: str) -> List[PageEntry]:
        """Walk all <a> tags, resolve URLs, deduplicate, return ordered list."""
        pages: List[PageEntry] = []
        seen: Set[str] = set()
        order = 0
        for a in container.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("#", "mailto:", "javascript:")):
                continue
            bare = href.split("#")[0].strip()
            if not bare:
                continue
            full = urllib.parse.urljoin(page_url, bare)
            # normalise directory URLs
            if not full.endswith((".html", ".htm")):
                full = full.rstrip("/") + "/index.html"
            if not self._is_internal(full) or full in seen:
                continue
            seen.add(full)
            depth = self._link_depth(a)
            pages.append(PageEntry(url=full, title=a.get_text().strip(),
                                   depth=depth, order=order))
            order += 1
        return pages

    def _link_depth(self, a_tag: Tag) -> int:
        li = a_tag.find_parent("li")
        if li:
            for cls in li.get("class", []):
                m = re.match(r"toctree-l(\d+)", cls)
                if m:
                    return int(m.group(1)) - 1
        depth = 0
        p = a_tag.parent
        while p and isinstance(p, Tag):
            if p.name == "ul":
                depth += 1
            p = p.parent
        return max(0, depth - 1)

    def _content_area(self, soup: BeautifulSoup) -> Optional[Tag]:
        for sel in [
            ("div", {"class_": "body"}),
            ("div", {"role": "main"}),
            ("div", {"class_": "document"}),
            ("main", {}),
            ("article", {}),
        ]:
            el = soup.find(sel[0], **sel[1])
            if el:
                return el
        return soup.find("body")

    def _is_internal(self, url: str) -> bool:
        # IMPROVED: accept URLs under either doc_root or base_url
        return (url.startswith(self.doc_root_url)
                or url.startswith(self.base_url))

    @staticmethod
    def _dedupe(pages: List[PageEntry]) -> List[PageEntry]:
        seen: Set[str] = set()
        out: List[PageEntry] = []
        for p in pages:
            if p.url not in seen:
                seen.add(p.url)
                out.append(p)
        return out


# =============================================================================
# Multi-Page Crawler  (IMPROVED — better next-link detection, BFS fallback)
# =============================================================================
class MultiPageDocCrawler:
    """
    Orchestrates:
        URL resolution  ->  page discovery  ->  per-page crawling  ->
        section extraction  ->  Markdown conversion  ->  file output
    """

    def __init__(self, *, url: str,
                 output_dir: Optional[str] = None,
                 delay: float              = DEFAULT_DELAY,
                 split_depth: int          = 2,
                 single_file: bool         = False,
                 download_images: bool     = False,
                 max_pages: int            = DEFAULT_MAX_PAGES,
                 list_pages_only: bool     = False):

        self.start_url       = url.rstrip("/")
        self.delay           = delay
        self.split_depth     = split_depth
        self.single_file     = single_file
        self.download_images = download_images
        self.max_pages       = max_pages
        self.list_pages_only = list_pages_only

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"),
            "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        })

        self.base_url, self.doc_root_url, self.slug = self._resolve_urls(url)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"{self.slug}_markdown")

        self.conv = MarkdownConverter(self.base_url)
        # IMPROVED: pass base_url so _is_internal can check both roots
        self.nav  = NavigationDiscovery(self.doc_root_url, self.base_url)

        self._pages_cache: Dict[str, BeautifulSoup] = {}
        self._doc_title: str = ""
        self._all_sections: OrderedDict[str, SectionInfo] = OrderedDict()

    # ------------------------------------------------------------------
    # URL resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_urls(url: str) -> Tuple[str, str, str]:
        parsed = urllib.parse.urlparse(url)
        path   = parsed.path

        if path.endswith(".html"):
            dir_path = path.rsplit("/", 1)[0] + "/"
        else:
            dir_path = path if path.endswith("/") else path + "/"

        base_url = f"{parsed.scheme}://{parsed.netloc}{dir_path}"

        CONTENT_DIRS = {"sections", "chapters", "parts",
                        "pages", "content", "modules", "api"}
        parts = dir_path.strip("/").split("/")
        root_parts = list(parts)
        while root_parts and root_parts[-1].lower() in CONTENT_DIRS:
            root_parts.pop()
        doc_root_url = (f"{parsed.scheme}://{parsed.netloc}/"
                        + "/".join(root_parts) + "/")

        SKIP = {"cuda", "latest", "stable", "current", "dev",
                "docs", "sections", "chapters", "parts"}
        candidates = [p for p in root_parts if p.lower() not in SKIP and p]
        slug = candidates[-1] if candidates else "nvidia-docs"

        return base_url, doc_root_url, slug

    # ------------------------------------------------------------------
    # HTTP
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
                    time.sleep(RETRY_WAIT * attempt * 2)
                elif code >= 500:
                    time.sleep(RETRY_WAIT * attempt)
                else:
                    log.error("  HTTP %d - skipping %s", code, url)
                    return None
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout):
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
    # Index-page detection
    # ------------------------------------------------------------------
    def _find_index_url(self) -> str:
        candidates = [
            self.doc_root_url + "index.html",
            self.doc_root_url,
            self.base_url + "index.html",
            self.base_url,
            self.start_url,
        ]
        seen: Set[str] = set()
        for url in candidates:
            if url in seen:
                continue
            seen.add(url)
            try:
                r = self.session.head(url, timeout=15, allow_redirects=True)
                if r.status_code == 200:
                    log.info("  Index page: %s", url)
                    return url
            except requests.RequestException:
                continue
        log.info("  Falling back to start URL as index")
        return self.start_url

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    def _detect_title(self, soup: BeautifulSoup) -> str:
        tag = soup.find("title")
        if tag:
            raw = tag.get_text().strip()
            raw = re.sub(r"\s*[—–-]\s*.*documentation.*$", "", raw, flags=re.I)
            raw = re.sub(r"\s*[—–-]\s*NVIDIA.*$", "", raw, flags=re.I)
            if raw:
                return raw.strip()
        h1 = soup.find("h1")
        if h1:
            return self.conv._heading_text(h1)
        return self.slug.replace("-", " ").title()

    # ------------------------------------------------------------------
    # Content root on a page
    # ------------------------------------------------------------------
    @staticmethod
    def _content_root(soup: BeautifulSoup) -> Tag:
        for sel in [
            {"name": "div", "attrs": {"class": "body"}},
            {"name": "div", "attrs": {"role": "main"}},
            {"name": "div", "attrs": {"class": "document"}},
            {"name": "main"},
            {"name": "article"},
        ]:
            el = soup.find(sel["name"], **sel.get("attrs", {}))
            if el:
                return el
        return soup.find("body") or soup

    # ------------------------------------------------------------------
    # Next-link detection  (IMPROVED — comprehensive pattern matching)
    # ------------------------------------------------------------------
    def _find_next_link(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Find the 'next page' link across various Sphinx theme formats:
          - <link rel="next" href="...">               (standard Sphinx)
          - <a rel="next" href="...">                   (RTD theme)
          - <a class="next-page" href="...">            (Furo theme)
          - <a class="right-next" href="...">           (Furo theme alt)
          - <a class="next" href="...">                 (generic)
          - footer <a> with 'next' in class             (fallback)
        """
        # 1. <link rel="next"> in <head>
        nxt = soup.find("link", rel="next")
        if nxt and nxt.get("href"):
            return nxt

        # 2. <a rel="next">
        nxt = soup.find("a", rel="next")
        if nxt and nxt.get("href"):
            return nxt

        # 3. Furo: <a class="next-page">
        nxt = soup.find("a", class_="next-page")
        if nxt and nxt.get("href"):
            return nxt

        # 4. Furo: <a class="right-next">
        nxt = soup.find("a", class_="right-next")
        if nxt and nxt.get("href"):
            return nxt

        # 5. Generic class="next"
        nxt = soup.find("a", class_="next")
        if nxt and nxt.get("href"):
            return nxt

        # 6. RTD: class="btn-next"
        nxt = soup.find("a", class_="btn-next")
        if nxt and nxt.get("href"):
            return nxt

        # 7. Footer-based: any <a> in <footer> with "next" in a CSS class
        for footer in soup.find_all("footer"):
            for a in footer.find_all("a", href=True):
                cls = a.get("class", [])
                if any("next" in c.lower() for c in cls):
                    return a

        # 8. Prev/next area (Furo/PyData): <div class="prev-next-area">
        #    or similar containers
        for container_cls in ("prev-next-area", "prev-next-bottom",
                              "prev-next-info", "rst-footer-buttons"):
            container = soup.find("div", class_=container_cls)
            if not container:
                continue
            # The "next" link is usually the last <a> or the one on the right
            links = container.find_all("a", href=True)
            if links:
                # Heuristic: the next link is the last one in the container
                return links[-1]

        return None

    # ------------------------------------------------------------------
    # Page discovery
    # ------------------------------------------------------------------
    def _discover_pages(self) -> List[PageEntry]:
        index_url  = self._find_index_url()
        index_soup = self._get_soup(index_url)
        if not index_soup:
            log.error("  Cannot fetch index page")
            return [PageEntry(url=self.start_url, title="", depth=0, order=0)]

        self._doc_title = self._detect_title(index_soup)
        log.info("  Document: %s", self._doc_title)

        # ---- Strategy 1: sidebar / toctree from index --------------------
        pages = self.nav.discover(index_soup, index_url)

        # ---- Strategy 2: if start URL differs, also try discovering ------
        if (len(pages) <= 1 and self.start_url != index_url):
            alt_soup = self._get_soup(self.start_url)
            if alt_soup:
                more = self.nav.discover(alt_soup, self.start_url)
                if len(more) > len(pages):
                    pages = more

        # ---- Strategy 3: follow rel="next" chain -------------------------
        if len(pages) <= 1:
            log.info("  Trying next-link chain from index ...")
            chain = self._follow_next_links(index_soup, index_url)
            if len(chain) > len(pages):
                pages = chain

        # ---- Strategy 4: BFS crawl  (NEW — robust fallback) -------------
        if len(pages) <= 1:
            log.info("  Trying BFS-based page discovery ...")
            bfs = self._bfs_discover(index_url, index_soup)
            if len(bfs) > len(pages):
                pages = bfs

        # ---- Strategy 5: BFS from start URL if different -----------------
        if len(pages) <= 1 and self.start_url != index_url:
            start_soup = self._get_soup(self.start_url)
            if start_soup:
                log.info("  Trying BFS from start URL ...")
                bfs2 = self._bfs_discover(self.start_url, start_soup)
                if len(bfs2) > len(pages):
                    pages = bfs2

        # ---- ensure the index page itself is present ---------------------
        urls = {p.url for p in pages}
        if index_url not in urls:
            pages.insert(0, PageEntry(url=index_url,
                                      title=self._doc_title,
                                      depth=0, order=-1))

        # ---- deduplicate & cap -------------------------------------------
        pages = NavigationDiscovery._dedupe(pages)
        if len(pages) > self.max_pages:
            log.warning("  Capping %d -> %d pages", len(pages), self.max_pages)
            pages = pages[:self.max_pages]

        log.info("  %d page(s) to crawl", len(pages))
        return pages

    def _follow_next_links(self, start_soup: BeautifulSoup,
                           start_url: str) -> List[PageEntry]:
        """Walk the rel='next' / class='next-page' chain from *start_url*."""
        pages = [PageEntry(url=start_url,
                           title=self._detect_title(start_soup),
                           depth=0, order=0)]
        seen = {start_url}
        cur_soup, cur_url = start_soup, start_url

        while len(pages) < self.max_pages:
            # IMPROVED: use comprehensive next-link finder
            nxt = self._find_next_link(cur_soup)
            if not nxt:
                break
            href = nxt.get("href", "")
            if not href:
                break
            nxt_url = urllib.parse.urljoin(cur_url, href.split("#")[0])
            if nxt_url in seen:
                break
            # Verify the next URL is still within this doc
            if not self.nav._is_internal(nxt_url):
                break
            seen.add(nxt_url)
            nxt_soup = self._get_soup(nxt_url)
            if not nxt_soup:
                break
            pages.append(PageEntry(url=nxt_url,
                                   title=self._detect_title(nxt_soup),
                                   depth=0, order=len(pages)))
            cur_soup, cur_url = nxt_soup, nxt_url

        log.info("    -> %d pages via next-link chain", len(pages))
        return pages if len(pages) > 1 else []

    # ------------------------------------------------------------------
    # BFS-based recursive page discovery  (NEW)
    # ------------------------------------------------------------------
    def _bfs_discover(self, start_url: str,
                      start_soup: BeautifulSoup) -> List[PageEntry]:
        """
        Breadth-first crawl from *start_url*, following every internal .html
        link found on each page.  This is the most robust fallback for when
        sidebar detection and next-link chains both fail.
        """
        pages = [PageEntry(url=start_url,
                           title=self._detect_title(start_soup),
                           depth=0, order=0)]
        seen: Set[str] = {start_url}
        queue: deque = deque([(start_url, 0)])
        order = 1
        # Cache start soup
        self._pages_cache.setdefault(start_url, start_soup)

        while queue and len(pages) < self.max_pages:
            cur_url, depth = queue.popleft()
            cur_soup = self._get_soup(cur_url)
            if not cur_soup:
                continue

            for a in cur_soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith(("#", "mailto:", "javascript:")):
                    continue
                bare = href.split("#")[0].strip()
                if not bare:
                    continue
                full = urllib.parse.urljoin(cur_url, bare)
                # Normalise directory URLs to index.html
                if not full.endswith((".html", ".htm")):
                    # Only normalise if it looks like a directory path
                    parsed = urllib.parse.urlparse(full)
                    if parsed.path.endswith("/"):
                        full = full.rstrip("/") + "/index.html"
                    else:
                        continue  # skip non-HTML links (images, pdfs, etc.)
                if full in seen:
                    continue
                if not self.nav._is_internal(full):
                    continue
                seen.add(full)
                title = a.get_text().strip()
                pages.append(PageEntry(url=full, title=title,
                                       depth=depth + 1, order=order))
                order += 1
                # Don't BFS too deeply to avoid crawling unrelated docs
                if depth < 2:
                    queue.append((full, depth + 1))

        log.info("    -> %d pages via BFS discovery", len(pages))
        return pages if len(pages) > 1 else []

    # ------------------------------------------------------------------
    # Section extraction (per page)
    # ------------------------------------------------------------------
    def _extract_sections(self, soup: BeautifulSoup,
                          page_url: str) -> OrderedDict:
        root = self._content_root(soup)
        sections: OrderedDict[str, SectionInfo] = OrderedDict()

        candidates = root.find_all("section", recursive=True)
        if not candidates:
            candidates = root.find_all("div", class_="section", recursive=True)

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
                    or "section" in p.get("class", [])
                ):
                    depth += 1
                    if p.get("id") and parent_id is None:
                        parent_id = p["id"]
                p = getattr(p, "parent", None)

            sections[sid] = SectionInfo(
                id=sid, title=htxt, section_number=snum,
                heading_level=hlvl, element=sec,
                page_url=page_url,
                parent_id=parent_id, depth=depth,
            )

        for sid, si in sections.items():
            if si.parent_id and si.parent_id in sections:
                sections[si.parent_id].children_ids.append(sid)

        return sections

    def _fallback_section(self, soup: BeautifulSoup,
                          page_url: str,
                          title: str) -> Optional[SectionInfo]:
        root = self._content_root(soup)
        if not root.get_text(strip=True):
            return None
        sid = re.sub(r"[^\w-]", "-", title.lower())[:60]
        sid = re.sub(r"-{2,}", "-", sid).strip("-") or "page-content"
        return SectionInfo(
            id=sid, title=title, section_number="",
            heading_level=1, element=root,
            page_url=page_url, depth=0,
        )

    # ------------------------------------------------------------------
    # Crawl all pages
    # ------------------------------------------------------------------
    def _crawl_all(self, page_entries: List[PageEntry]) -> OrderedDict:
        all_sec: OrderedDict[str, SectionInfo] = OrderedDict()
        g_order = 0

        for i, page in enumerate(page_entries, 1):
            log.info("  [%d/%d] %s", i, len(page_entries), page.url)
            soup = self._get_soup(page.url)
            if not soup:
                log.warning("    x  failed, skipping")
                continue

            page_base = page.url.rsplit("/", 1)[0] + "/"
            self.conv.base_url = page_base.rstrip("/")

            secs = self._extract_sections(soup, page.url)

            if secs:
                for sid, si in secs.items():
                    uid = sid
                    if uid in all_sec:
                        slug = re.sub(r"[^\w]", "-",
                                      page.url.rsplit("/", 1)[-1]
                                      .replace(".html", ""))
                        uid = f"{slug}--{sid}"
                        si.id = uid
                    si.global_order = g_order
                    g_order += 1
                    all_sec[uid] = si
                log.info("    -> %d sections", len(secs))
            else:
                title = page.title or self._detect_title(soup)
                fb = self._fallback_section(soup, page.url, title)
                if fb:
                    uid = fb.id
                    if uid in all_sec:
                        uid = f"page-{i}--{uid}"
                        fb.id = uid
                    fb.global_order = g_order
                    g_order += 1
                    all_sec[uid] = fb
                    log.info("    -> 1 section (whole-page fallback)")
                else:
                    log.info("    -> (empty)")

        self.conv.base_url = self.base_url.rstrip("/")
        return all_sec

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
    # Front-matter
    # ------------------------------------------------------------------
    def _front_matter(self, title: str, section_num: str, url: str) -> str:
        safe = title.replace('"', '\\"')
        return (f"---\n"
                f'title: "{safe}"\n'
                f'section: "{section_num}"\n'
                f'source: "{url}"\n'
                f"---\n\n")

    # ------------------------------------------------------------------
    # Image downloading
    # ------------------------------------------------------------------
    def _download_images_from_md(self, md: str, section_id: str) -> str:
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        counter = [0]

        def _repl(m):
            alt, url = m.group(1), m.group(2)
            if url.startswith(("data:", "images/")):
                return m.group(0)
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
            except Exception:
                return m.group(0)
            ct  = resp.headers.get("Content-Type", "")
            ext = ".png"
            for mime, e in [("jpeg", ".jpg"), ("gif", ".gif"),
                            ("svg", ".svg"), ("webp", ".webp")]:
                if mime in ct:
                    ext = e; break
            counter[0] += 1
            fn = f"{re.sub(r'[^\\w-]', '_', section_id)}_{counter[0]}{ext}"
            (img_dir / fn).write_bytes(resp.content)
            return f"![{alt}](images/{fn})"

        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _repl, md)

    # ------------------------------------------------------------------
    # Index file
    # ------------------------------------------------------------------
    def _save_index(self, to_save: List[SectionInfo]):
        title = self._doc_title or self.slug.replace("-", " ").title()
        lines = [
            "---",
            f"title: {title} - Index",
            f"source: {self.doc_root_url}",
            f"crawled: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "---", "",
            f"# {title}", "",
            f"Source: <{self.doc_root_url}>", "",
            "## Table of Contents", "",
        ]
        for i, si in enumerate(to_save):
            indent = "  " * si.depth
            fn = self._filename(si, i)
            label = f"{si.section_number}. " if si.section_number else ""
            clean = re.sub(r"^[\d.]+\s*", "", si.title)
            lines.append(f"{indent}- [{label}{clean}](./{fn})")
        lines.append("")
        f = self.output_dir / "00_index.md"
        f.write_text("\n".join(lines), encoding="utf-8")
        log.info("  Saved index -> %s", f.name)

    # ------------------------------------------------------------------
    # Output: single file
    # ------------------------------------------------------------------
    def _write_single(self):
        parts: list[str] = []
        for si in self._all_sections.values():
            pb = si.page_url.rsplit("/", 1)[0] + "/"
            self.conv.base_url = pb.rstrip("/")
            parts.append(self.conv.convert_section_body(
                si.element, include_children=False))
        self.conv.base_url = self.base_url.rstrip("/")
        md = "\n".join(parts)
        slug = re.sub(r"[^\w\-]", "-", self.slug).strip("-")
        header = (
            "---\n"
            f"title: {self._doc_title} (complete)\n"
            f"source: {self.doc_root_url}\n"
            f"crawled: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "---\n\n"
        )
        out = self.output_dir / f"{slug}-complete.md"
        out.write_text(header + md, encoding="utf-8")
        log.info("  -> %s  (%.1f KB)", out.name, out.stat().st_size / 1024)

    # ------------------------------------------------------------------
    # Output: split files
    # ------------------------------------------------------------------
    def _write_split(self):
        to_save: list[SectionInfo] = []
        for si in self._all_sections.values():
            if self.split_depth == 0 or si.depth <= self.split_depth:
                to_save.append(si)
        if not to_save:
            to_save = list(self._all_sections.values())

        self._save_index(to_save)
        save_ids = {s.id for s in to_save}
        total = len(to_save)

        for idx, si in enumerate(to_save, 1):
            pb = si.page_url.rsplit("/", 1)[0] + "/"
            self.conv.base_url = pb.rstrip("/")

            children_sep = any(cid in save_ids for cid in si.children_ids)
            md = self.conv.convert_section_body(
                si.element, include_children=not children_sep)
            if not md.strip():
                continue
            if self.download_images:
                md = self._download_images_from_md(md, si.id)

            src = f"{si.page_url}#{si.id}"
            fm  = self._front_matter(si.title, si.section_number, src)
            fn  = self._filename(si, idx)
            out = self.output_dir / fn
            out.write_text(fm + md, encoding="utf-8")
            kb = out.stat().st_size / 1024
            log.info("  [%d/%d] %s  (%.1f KB)", idx, total, fn, kb)

        self.conv.base_url = self.base_url.rstrip("/")
        log.info("  Total: %d files + index", total)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> bool:
        log.info("=" * 60)
        log.info("NVIDIA Multi-Page Documentation Crawler")
        log.info("=" * 60)
        log.info("Start URL  : %s", self.start_url)
        log.info("Base URL   : %s", self.base_url)
        log.info("Doc Root   : %s", self.doc_root_url)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output     : %s", self.output_dir.resolve())

        # ---- 1. discover pages -------------------------------------------
        log.info("\n[1/3] Discovering pages ...")
        pages = self._discover_pages()

        # ---- list-only mode ----------------------------------------------
        if self.list_pages_only:
            print(f"\n{'=' * 60}")
            print(f"Document : {self._doc_title}")
            print(f"Pages    : {len(pages)}")
            print(f"{'=' * 60}")
            for i, p in enumerate(pages, 1):
                indent = "  " * p.depth
                print(f"  {i:3d}. {indent}{p.title or '(untitled)'}")
                print(f"       {p.url}")
            return True

        # ---- 2. crawl & extract ------------------------------------------
        log.info("\n[2/3] Crawling pages & extracting sections ...")
        self._all_sections = self._crawl_all(pages)
        if not self._all_sections:
            log.error("  No sections found - aborting.")
            return False
        log.info("  Total: %d sections from %d page(s)",
                 len(self._all_sections), len(pages))

        # ---- 3. save -----------------------------------------------------
        log.info("\n[3/3] Converting to Markdown ...")
        if self.single_file:
            self._write_single()
        else:
            self._write_split()

        log.info("\n" + "=" * 60)
        log.info("Done!  Files in: %s", self.output_dir.resolve())
        log.info("=" * 60)
        return True


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Crawl multi-page NVIDIA Sphinx documentation -> Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
  %(prog)s https://docs.nvidia.com/cuda/cutile-python/index.html
  %(prog)s https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html
  %(prog)s <URL> -o my_docs --single-file
  %(prog)s <URL> --list-pages
  %(prog)s <URL> --max-pages 50 --split-depth 3
        """,
    )
    ap.add_argument("url",
                    help="URL of any page in the documentation")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="output directory  (default: auto from doc name)")
    ap.add_argument("-d", "--delay", type=float, default=DEFAULT_DELAY,
                    help=f"seconds between requests  (default: {DEFAULT_DELAY})")
    ap.add_argument("--split-depth", type=int, default=2,
                    help="section nesting depth for file splitting (0=all)")
    ap.add_argument("--single-file", action="store_true",
                    help="write one combined Markdown file")
    ap.add_argument("--download-images", action="store_true",
                    help="download images to local images/ directory")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                    help=f"stop after N pages  (default: {DEFAULT_MAX_PAGES})")
    ap.add_argument("--list-pages", action="store_true",
                    help="discover & list pages only, do not crawl content")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="debug-level logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ok = MultiPageDocCrawler(
        url=args.url,
        output_dir=args.output_dir,
        delay=args.delay,
        split_depth=args.split_depth,
        single_file=args.single_file,
        download_images=args.download_images,
        max_pages=args.max_pages,
        list_pages_only=args.list_pages,
    ).run()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
