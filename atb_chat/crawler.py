"""
atb_chat.crawler : Crawl AllTheBacteria resources into JSONL.

Called via:  atb-chat crawl

Sources:
  preprint   biorxiv PDF: one chunk per paragraph per page
  docs       ReadTheDocs pages: one entry per page (full text)
  github     repo tree + issues: one chunk per paragraph
"""

import json
import logging
import re
import subprocess
import tempfile
import time
import pymupdf
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

try:
    from markdownify import markdownify as md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False

log = logging.getLogger("atb-chat-crawler")

# Target URLs
# Currently hardcoded, but could be made configurable in the future if needed
PREPRINT_URL = "https://www.biorxiv.org/content/10.1101/2024.03.08.584059v3.full.pdf"
READTHEDOCS_BASE = "https://allthebacteria.readthedocs.io/en/latest/"
GITHUB_OWNER = "AllTheBacteria"
GITHUB_REPO = "AllTheBacteria"
GITHUB_API = "https://api.github.com"

ALL_SOURCES = ["preprint", "docs", "github"]

# GitHub file filters (matches repo languages: Python, Shell, Nextflow, Perl)
CODE_EXTENSIONS = frozenset({
    ".py",
    ".sh", ".bash",
    ".nf", ".config",
    ".pl", ".pm",
    ".md", ".rst",
})


# ── helpers ──────────────────────────────────────────────────────────────────
def _curl(url):
    """Fetch URL via curl."""
    r = subprocess.run(
        ["curl", "-sL", url],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {r.stderr}")
    return r.stdout


def _curl_json(url):
    """Fetch JSON from GitHub API via curl."""
    cmd = ["curl", "-sL", "-H", "Accept: application/vnd.github+json", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {r.stderr}")
    return json.loads(r.stdout)


def _html_to_text(html):
    """Convert HTML to plain text, preserving structure."""
    if HAS_MARKDOWNIFY:
        text = md(html, heading_style="ATX", strip=["img", "script", "style"])
    else:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _save_jsonl(entries, path):
    """Write list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("  ✓ %d entries → %s", len(entries), path)


# ── 1. Preprint manuscript ──────────────────────────────────────────────────
def crawl_preprint():
    """Download preprint PDF via wget, extract text with pymupdf."""
    

    log.info("   Crawling and processing preprint")
    print(f"   Preprint")
    print(f"   Downloading PDF from {PREPRINT_URL}")

    t0 = time.time()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    try:
        subprocess.run(["wget", "-q", "-O", tmp.name, PREPRINT_URL], check=True)
        doc = pymupdf.Document(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    print(f"   Downloaded ({len(doc)} pages). Extracting text ...")

    entries = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if not text.strip():
            continue
        for idx, para in enumerate(re.split(r"\n\n+", text)):
            if not para.strip():
                continue
            entries.append({
                "id": f"preprint_p{page_num + 1}_c{idx}",
                "text": para.strip(),
                "metadata": {
                    "source": "preprint",
                    "url": PREPRINT_URL,
                    "title": "AllTheBacteria preprint (biorxiv 2024.03.08.584059v3)",
                    "section": f"page_{page_num + 1}",
                    "chunk_idx": idx,
                },
            })
    page_count = len(doc)
    doc.close()

    elapsed = time.time() - t0
    print(f"   {len(entries)} chunks from {page_count} pages ({elapsed:.1f}s)")
    log.info("  preprint → %d entries", len(entries))
    return entries


# ── 2. Docs from ReadTheDocs ────────────────────────────────────────────────
def _discover_doc_pages():
    """Scrape ReadTheDocs sidebar for all doc page URLs."""
    pages = set()
    html = _curl(READTHEDOCS_BASE)
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and READTHEDOCS_BASE in href:
            pages.add(href.split("#")[0])
        elif href.endswith(".html") and not href.startswith("#"):
            pages.add(urljoin(READTHEDOCS_BASE, href).split("#")[0])
    return sorted(pages)


def crawl_docs():
    """Crawl all ReadTheDocs pages. One entry per page (full text, no chunking)."""
    log.info("Crawling and processing docs from ReadTheDocs")
    print(f"   Docs from ReadTheDocs")
    print(f"   Discovering pages from {READTHEDOCS_BASE}")

    t0 = time.time()
    pages = _discover_doc_pages()
    print(f"   Found {len(pages)} pages. Crawling ...")
    log.info("  discovered %d pages", len(pages))

    entries = []
    for i, page_url in enumerate(pages, 1):
        try:
            html = _curl(page_url)
        except Exception as e:
            log.warning("  SKIP %s: %s", page_url, e)
            print(f"   ⚠ [{i}/{len(pages)}] SKIP {page_url.split('/')[-1]}: {e}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        main = (
            soup.find(role="main")
            or soup.find("div", class_="document")
            or soup.find("div", class_="body")
            or soup.find("main")
            or soup
        )
        for nav in main.find_all(["nav", "footer"]):
            nav.decompose()

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else urlparse(page_url).path
        slug = page_url.split("/")[-1].replace(".html", "") or "index"
        text = _html_to_text(str(main))

        entries.append({
            "id": f"docs_{slug}",
            "text": text,
            "metadata": {
                "source": "docs",
                "url": page_url,
                "title": title,
            },
        })
        print(f"   [{i}/{len(pages)}] {slug} → \"{title}\" ({len(text):,} chars)")
        log.info("  %s → \"%s\"", slug, title)

    elapsed = time.time() - t0
    print(f"   ✓ {len(entries)} pages crawled ({elapsed:.1f}s)")
    log.info("  docs total → %d entries", len(entries))
    return entries


# ── 3. GitHub tree and issues ───────────────────────────────────────────────
def _crawl_github_tree():
    """Crawl repo file tree for READMEs and code files."""
    log.info("  GitHub tree crawl")
    print(f"   Fetching file tree ...")
    url = f"{GITHUB_API}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/main?recursive=1"
    tree = _curl_json(url)

    all_blobs = [
        item for item in tree.get("tree", [])
        if item["type"] == "blob" and (
            Path(item["path"]).name.lower() in ("readme.md", "readme.rst", "readme.txt", "readme")
            or Path(item["path"]).suffix.lower() in CODE_EXTENSIONS
        )
    ]
    print(f"   Found {len(all_blobs)} matching files (of {len(tree.get('tree', []))} total). Downloading ...")

    entries = []
    for i, item in enumerate(all_blobs, 1):
        path = item["path"]
        name = Path(path).name.lower()
        is_readme = name in ("readme.md", "readme.rst", "readme.txt", "readme")

        raw_url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/{path}"
        blob_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/blob/main/{path}"
        r = subprocess.run(
            ["curl", "-sL", raw_url],
            capture_output=True, text=True, timeout=30,
        )
        content = r.stdout
        if not content.strip():
            continue

        src = "github_readme" if is_readme else "github_code"
        slug = path.replace("/", "_").replace(".", "_")
        file_chunks = 0
        for idx, para in enumerate(re.split(r"\n\n+", content)):
            if not para.strip():
                continue
            entries.append({
                "id": f"{src}_{slug}_c{idx}",
                "text": para.strip(),
                "metadata": {
                    "source": src,
                    "url": blob_url,
                    "title": path,
                    "chunk_idx": idx,
                },
            })
            file_chunks += 1
        print(f"   [{i}/{len(all_blobs)}] {path} → {file_chunks} chunks")
        log.info("    %s → %d entries", path, file_chunks)

    return entries


def _crawl_github_issues():
    """Crawl all open+closed issues with comments."""
    log.info("Crawling and processing GitHub issues")
    print(f"   Fetching issues ...")
    entries = []
    page = 1
    issue_count = 0

    while True:
        url = (
            f"{GITHUB_API}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues"
            f"?state=all&per_page=100&page={page}&sort=updated&direction=desc"
        )
        issues = _curl_json(url)
        if not issues:
            break

        # In case GitHub API returns a dict with "message" key on error (e.g. rate limit), instead of a list of issues
        if not isinstance(issues, list):
            print("Unexpected response from GitHub API:", issues)
            break

        for iss in issues:
            if "pull_request" in iss:
                continue

            num = iss["number"]
            title = iss.get("title", "")
            body = iss.get("body", "") or ""
            state = iss.get("state", "")
            labels = [lb["name"] for lb in iss.get("labels", [])]
            created = iss.get("created_at", "")
            html_url = iss.get("html_url", "")
            n_comments = iss.get("comments", 0)

            parts = [
                f"# Issue #{num}: {title}",
                f"State: {state}  |  Labels: {', '.join(labels) or 'none'}  |  Created: {created}",
                "",
                body,
            ]

            if n_comments > 0:
                try:
                    comments = _curl_json(iss["comments_url"])
                    for c in comments:
                        who = c.get("user", {}).get("login", "?")
                        parts.append(f"\n---\n**{who}:**\n{c.get('body', '')}")
                except Exception:
                    pass

            full_text = "\n".join(parts)
            issue_chunks = 0
            for idx, para in enumerate(re.split(r"\n\n+", full_text)):
                if not para.strip():
                    continue
                entries.append({
                    "id": f"github_issue_{num}_c{idx}",
                    "text": para.strip(),
                    "metadata": {
                        "source": "github_issue",
                        "url": html_url,
                        "title": f"Issue #{num}: {title}",
                        "chunk_idx": idx,
                    },
                })
                issue_chunks += 1

            issue_count += 1
            print(f"   #{num} \"{title}\" ({issue_chunks} chunks, {n_comments} comments)")

        log.info("    page %d → %d issues", page, len(issues))
        page += 1
        if len(issues) < 100:
            break

    print(f"   Processed {issue_count} issues")
    return entries


def crawl_github():
    """Crawl GitHub repo tree and issues."""
    log.info("Crawling and processing GitHub repo tree and issues")
    print(f"   GitHub ({GITHUB_OWNER}/{GITHUB_REPO})")

    t0 = time.time()
    tree_entries = _crawl_github_tree()
    print(f"   Tree: {len(tree_entries)} chunks")
    issue_entries = _crawl_github_issues()
    print(f"   Issues: {len(issue_entries)} chunks")
    total = tree_entries + issue_entries

    elapsed = time.time() - t0
    print(f"   ✓ {len(total)} total chunks ({elapsed:.1f}s)")
    log.info("  github total → %d entries (%d tree + %d issues)",
             len(total), len(tree_entries), len(issue_entries))
    return total


# ── orchestrator (called by cli.py) ─────────────────────────────────────────
SOURCE_DIRS = {
    "preprint": "preprint",
    "docs": "doc",
    "github": "code",
}


def run(args):
    """Entry point called by cli.py → run_crawl(args).

    Expected args attributes:
        args.sources    - list[str]
        args.crawl_dir  - str
        args.verbose    - bool
    """
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(getattr(args, "crawl-dir", "./guides"))
    sources = getattr(args, "sources", ALL_SOURCES)

    log.info("Starting ATB knowledge-base crawl")

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║           ATB Knowledge Base Crawler                     ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\nSources to crawl: {', '.join(sources)}")
    print(f"Output directory:  {output_dir.resolve()}")
    log.info("  sources : %s", ", ".join(sources))
    log.info("  output  : %s", output_dir.resolve())

    t_start = time.time()
    all_entries = []

    for src in sources:
        try:
            if src == "preprint":
                entries = crawl_preprint()
            elif src == "docs":
                entries = crawl_docs()
            elif src == "github":
                entries = crawl_github()
            else:
                log.warning("Unknown source: %s", src)
                continue
        except Exception as exc:
            log.error("'%s' crawl failed: %s", src, exc, exc_info=True)
            print(f"   '{src}' crawl failed: {exc}")
            continue

        sub = SOURCE_DIRS.get(src, src)
        out_path = output_dir / sub / f"{src}.jsonl"
        _save_jsonl(entries, out_path)
        all_entries.extend(entries)

    # manifest
    sources_count = {}
    for e in all_entries:
        s = e["metadata"]["source"]
        sources_count[s] = sources_count.get(s, 0) + 1

    manifest = {
        "project": "AllTheBacteria",
        "total_entries": len(all_entries),
        "sources": sources_count,
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t_start
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print(f"║  DONE — {len(all_entries):,} entries in {elapsed:.1f}s")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\nOutput files:")
    for f in sorted(output_dir.rglob("*.jsonl")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(output_dir)}  ({size_kb:.1f} KB)")
    print(f"  manifest.json")
    print(f"\nNext step: atb-chat ingest --db-dir <path>")

    log.info("")
    log.info("DONE – %d total entries across %d source(s) in %.1fs",
             len(all_entries), len(sources), elapsed)