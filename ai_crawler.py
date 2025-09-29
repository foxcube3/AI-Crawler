import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from urllib.parse import urlparse, urljoin
import urllib.robotparser as robotparser
from io import BytesIO
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import trafilatura

# Optional, used for query-based discovery without API keys
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None  # fallback if not available

# Document extraction deps
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from ebooklib import epub
except Exception:
    epub = None

# Embeddings / vector store
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception:
    np = None
    faiss = None
    SentenceTransformer = None


DEFAULT_USER_AGENTS = [
    # Chrome Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    # Chrome Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36 Edg/116.0",
]


class UserAgentRotator:
    def __init__(self, user_agents: List[str]):
        self.user_agents = user_agents or DEFAULT_USER_AGENTS
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            ua = self.user_agents[self._i % len(self.user_agents)]
            self._i += 1
            return ua


class PerDomainRateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str):
        if self.delay <= 0:
            return
        with self._lock:
            now = time.time()
            last = self._last.get(domain, 0.0)
            wait = self.delay - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._last[domain] = time.time()


@dataclass
class CrawlResult:
    url: str
    title: Optional[str]
    text_path: Optional[str]
    status: str
    domain: str
    error: Optional[str] = None


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name[:120].strip("-") or "page"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch(url: str, timeout: int, user_agent: str) -> Tuple[Optional[requests.Response], Optional[str]]:
    headers = {"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def guess_title_from_html(html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else None
    except Exception:
        return None


def extract_text_from_html(html: str, url: str) -> Optional[str]:
    try:
        downloaded = trafilatura.extract(html, url=url, include_comments=False)
        if downloaded and downloaded.strip():
            return downloaded.strip()
    except Exception:
        pass
    try:
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    return None


def extract_text_from_pdf(data: bytes) -> Optional[str]:
    if pdf_extract_text is None:
        return None
    try:
        return pdf_extract_text(BytesIO(data)) or None
    except Exception:
        return None


def extract_text_from_docx(data: bytes) -> Optional[str]:
    if docx is None:
        return None
    try:
        document = docx.Document(BytesIO(data))
        paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paragraphs) or None
    except Exception:
        return None


def extract_text_from_epub(data: bytes) -> Optional[str]:
    if epub is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=True) as tf:
            tf.write(data)
            tf.flush()
            book = epub.read_epub(tf.name)
        texts: List[str] = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                try:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    t = soup.get_text(separator="\n")
                    if t and t.strip():
                        texts.append(t.strip())
                except Exception:
                    continue
        return "\n\n".join(texts) if texts else None
    except Exception:
        return None


def detect_and_extract(resp: requests.Response, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type or (not content_type and "<html" in resp.text[:1000].lower()):
        html = resp.text
        text = extract_text_from_html(html, url)
        title = guess_title_from_html(html)
        return text, title, html
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text = extract_text_from_pdf(resp.content)
        return text, None, None
    if "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type or url.lower().endswith(".docx"):
        text = extract_text_from_docx(resp.content)
        return text, None, None
    if "application/epub+zip" in content_type or url.lower().endswith(".epub"):
        text = extract_text_from_epub(resp.content)
        return text, None, None
    try:
        html = resp.text
        text = extract_text_from_html(html, url)
        title = guess_title_from_html(html)
        return text, title, html
    except Exception:
        return None, None, None


def save_text(text: str, url: str, output_dir: str, title: Optional[str]) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc
    slug_source = title or parsed.path or "page"
    slug = sanitize_filename(slug_source)
    dir_path = os.path.join(output_dir, domain)
    ensure_dir(dir_path)
    file_path = os.path.join(dir_path, f"{slug}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path


def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme.startswith("http"):
            links.append(abs_url)
    seen = set()
    unique = []
    for u in links:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def process_url(current_url: str, depth: int, output_dir: str, timeout: int, allow_external: bool, crawl_depth: int, ua_rotator: UserAgentRotator, rate_limiter: PerDomainRateLimiter) -> Tuple[CrawlResult, List[Tuple[str, int]]]:
    domain = urlparse(current_url).netloc
    rate_limiter.wait(domain)
    ua = ua_rotator.next()
    resp, err = fetch(current_url, timeout=timeout, user_agent=ua)
    if resp is None:
        return CrawlResult(url=current_url, title=None, text_path=None, status="fetch_error", domain=domain, error=err), []

    text, title, html = detect_and_extract(resp, current_url)
    if not text:
        result = CrawlResult(url=current_url, title=None, text_path=None, status="extraction_failed", domain=domain)
        nexts: List[Tuple[str, int]] = []
    else:
        text_path = save_text(text, current_url, output_dir, title)
        result = CrawlResult(url=current_url, title=title, text_path=text_path, status="ok", domain=domain)
        nexts = []

    # Expand links if HTML and depth allows
    if html and depth < crawl_depth:
        for nl in extract_links(html, current_url):
            if not allow_external and urlparse(nl).netloc != domain:
                continue
            nexts.append((nl, depth + 1))

    return result, nexts


def crawl(
    query: Optional[str],
    urls: Optional[List[str]],
    output_dir: str = "data",
    max_results: int = 10,
    crawl_depth: int = 1,
    max_pages_total: int = 50,
    timeout: int = 20,
    allow_external: bool = True,
    concurrency: int = 4,
    per_domain_delay: float = 0.5,
    user_agents: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    ensure_dir(output_dir)
    discovered: List[str] = []
    if query:
        discovered = discover_urls_by_query(query, max_results=max_results)
    if urls:
        discovered.extend(urls)

    seen = set()
    seeds: List[str] = []
    for u in discovered:
        if u not in seen:
            seeds.append(u)
            seen.add(u)

    results: List[CrawlResult] = []
    visited: set = set()
    queue: List[Tuple[str, int]] = [(u, 0) for u in seeds]

    ua_rotator = UserAgentRotator(user_agents or DEFAULT_USER_AGENTS)
    rate_limiter = PerDomainRateLimiter(delay=per_domain_delay)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures: Dict = {}
        while (queue or futures) and len(results) < max_pages_total:
            # Launch new tasks up to concurrency
            while queue and len(futures) < concurrency and len(results) + len(futures) < max_pages_total:
                current_url, depth = queue.pop(0)
                if current_url in visited:
                    continue
                visited.add(current_url)
                fut = executor.submit(process_url, current_url, depth, output_dir, timeout, allow_external, crawl_depth, ua_rotator, rate_limiter)
                futures[fut] = (current_url, depth)

            # Collect completed tasks
            if futures:
                for fut in as_completed(list(futures.keys())):
                    current_url, depth = futures.pop(fut)
                    try:
                        result, nexts = fut.result()
                        results.append(result)
                        for nl, nd in nexts:
                            if nl not in visited:
                                queue.append((nl, nd))
                    except Exception as e:
                        results.append(CrawlResult(url=current_url, title=None, text_path=None, status="error", domain=urlparse(current_url).netloc, error=f"{type(e).__name__}: {e}"))
                    # Break to allow launching more tasks in the outer loop
                    break

    index_path = os.path.join(output_dir, "index.json")
    index_data = [asdict(r) for r in results]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    return {"results": index_data, "index_path": index_path}


def discover_urls_by_query(query: str, max_results: int = 10) -> List[str]:
    urls: List[str] = []
    if DDGS is None:
        return urls
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                u = r.get("href") or r.get("url")
                if u:
                    urls.append(u)
    except Exception:
        pass
    return urls


# ---------- Vector store and Q&A ----------

def load_text_files(output_dir: str) -> List[Tuple[str, str]]:
    data: List[Tuple[str, str]] = []
    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn.endswith(".txt"):
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        txt = f.read()
                        if txt and txt.strip():
                            data.append((path, txt))
                except Exception:
                    continue
    return data


def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    parts: List[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    current = []
    length = 0
    for p in paragraphs:
        if length + len(p) + 2 <= max_chars:
            current.append(p)
            length += len(p) + 2
        else:
            if current:
                parts.append("\n\n".join(current))
            current = [p]
            length = len(p)
    if current:
        parts.append("\n\n".join(current))
    return parts


def build_vector_index(output_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, str]:
    if np is None or faiss is None or SentenceTransformer is None:
        raise RuntimeError("Vector store dependencies not available. Please install requirements.")
    docs = load_text_files(output_dir)
    if not docs:
        raise RuntimeError("No text files found to index.")

    model = SentenceTransformer(model_name)
    passages: List[str] = []
    meta: List[Dict] = []
    for path, text in docs:
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            passages.append(ch)
            meta.append({"path": path, "chunk": i})

    embeddings = model.encode(passages, normalize_embeddings=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    idx_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, idx_path)
    with open(os.path.join(output_dir, "passages.json"), "w", encoding="utf-8") as f:
        json.dump({"passages": passages, "meta": meta}, f, ensure_ascii=False)

    return {"index_path": idx_path, "passages_path": os.path.join(output_dir, "passages.json")}


def ask_question(output_dir: str, question: str, top_k: int = 5, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    if np is None or faiss is None or SentenceTransformer is None:
        raise RuntimeError("Vector store dependencies not available. Please install requirements.")
    idx_path = os.path.join(output_dir, "index.faiss")
    passages_path = os.path.join(output_dir, "passages.json")
    if not os.path.exists(idx_path) or not os.path.exists(passages_path):
        build_vector_index(output_dir, model_name=model_name)

    index = faiss.read_index(idx_path)
    with open(passages_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    passages = data["passages"]
    meta = data["meta"]

    model = SentenceTransformer(model_name)
    q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    hits = []
    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(passages):
            continue
        hits.append({
            "score": float(s),
            "passage": passages[i],
            "source": meta[i]["path"],
            "chunk": meta[i]["chunk"],
        })

    answer = hits[0]["passage"] if hits else ""
    return {"question": question, "answer": answer, "hits": hits}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Assistant: crawl the web (HTML/PDF/DOCX/EPUB) with concurrency & UA rotation, build vector index, and Q&A.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--query", type=str, help="Search query to discover pages (DuckDuckGo).")
    group.add_argument("--urls", nargs="+", help="Specific URLs to crawl.")
    parser.add_argument("--max-results", type=int, default=10, help="Max search results to crawl.")
    parser.add_argument("--crawl-depth", type=int, default=1, help="Depth for link expansion (can cross domains if allowed).")
    parser.add_argument("--max-pages-total", type=int, default=50, help="Global page cap for crawling.")
    parser.add_argument("--allow-external", action="store_true", help="Allow following links across different domains.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for text files and index.json.")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    parser.add_argument("--build-index", action="store_true", help="Build vector index over crawled corpus.")
    parser.add_argument("--ask", type=str, help="Ask a question over the built corpus.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve for Q&A.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent fetch workers.")
    parser.add_argument("--per-domain-delay", type=float, default=0.5, help="Minimum delay (seconds) between requests to the same domain.")
    parser.add_argument("--user-agents-file", type=str, help="Path to file containing user-agents (one per line).")
    return parser.parse_args()


def main():
    args = parse_args()
    user_agents = None
    if args.user_agents_file and os.path.exists(args.user_agents_file):
        try:
            with open(args.user_agents_file, "r", encoding="utf-8") as f:
                user_agents = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            user_agents = None

    if args.ask:
        res = ask_question(output_dir=args.output_dir, question=args.ask, top_k=args.top_k)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    if args.build_index:
        res = build_vector_index(output_dir=args.output_dir)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    res = crawl(
        query=args.query,
        urls=args.urls,
        output_dir=args.output_dir,
        max_results=args.max_results,
        crawl_depth=args.crawl_depth,
        max_pages_total=args.max_pages_total,
        timeout=args.timeout,
        allow_external=args.allow_external,
        concurrency=args.concurrency,
        per_domain_delay=args.per_domain_delay,
        user_agents=user_agents,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()