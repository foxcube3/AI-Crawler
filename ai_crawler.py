import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl
from io import BytesIO
import tempfile
import threading
import heapq
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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36 Edg/116.0",
]


TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_name", "gclid", "fbclid"}


def normalize_url(url: str, base: Optional[str] = None) -> Optional[str]:
    try:
        if base:
            url = urljoin(base, url)
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return None
        # strip fragment
        fragmentless = parsed._replace(fragment="")
        # lowercase host
        netloc = fragmentless.netloc.lower()
        # remove default ports
        if netloc.endswith(":80") and fragmentless.scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and fragmentless.scheme == "https":
            netloc = netloc[:-4]
        # normalize path (remove trailing slash for non-root)
        path = fragmentless.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        # strip tracking params
        q = fragmentless.query
        if q:
            params = [(k, v) for k, v in parse_qsl(q, keep_blank_values=True) if k not in TRACKING_PARAMS]
            query = "&".join([f"{k}={v}" if v != "" else k for k, v in params])
        else:
            query = ""
        normalized = urlunparse((fragmentless.scheme, netloc, path, fragmentless.params, query, ""))
        return normalized
    except Exception:
        return None


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
    def __init__(self, default_delay: float):
        self.default_delay = default_delay
        self._last: Dict[str, float] = {}
        self._overrides: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set_delay(self, domain: str, delay: float):
        with self._lock:
            self._overrides[domain] = max(0.0, delay)

    def get_delay(self, domain: str) -> float:
        with self._lock:
            return self._overrides.get(domain, self.default_delay)

    def wait(self, domain: str):
        delay = self.get_delay(domain)
        if delay <= 0:
            return
        with self._lock:
            now = time.time()
            last = self._last.get(domain, 0.0)
            wait = delay - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._last[domain] = time.time()


def fetch_with_retry(url: str, timeout: int, user_agent: str, max_retries: int, backoff_base: float, backoff_jitter: float) -> Tuple[Optional[requests.Response], Optional[str]]:
    headers = {"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"}
    attempt = 0
    while attempt <= max_retries:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp, None
        except Exception as e:
            if attempt == max_retries:
                return None, f"{type(e).__name__}: {e}"
            sleep = backoff_base * (2 ** attempt) + (backoff_jitter * (0.5 - time.time() % 1))
            time.sleep(max(0.1, sleep))
            attempt += 1
    return None, "UnknownError"


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
        norm = normalize_url(href, base=base_url)
        if norm:
            links.append(norm)
    seen = set()
    unique = []
    for u in links:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique

def matches_patterns(url: str, patterns: Optional[List[str]]) -> bool:
    if not patterns:
        return False
    for p in patterns:
        try:
            if re.search(p, url):
                return True
        except Exception:
            # Treat as simple substring if regex invalid
            if p in url:
                return True
    return False

def should_follow(url: str, current_domain: str, allow_external: bool, allowlist_global: Optional[List[str]], denylist_global: Optional[List[str]], allowlist_by_domain: Optional[Dict[str, List[str]]], denylist_by_domain: Optional[Dict[str, List[str]]]) -> bool:
    dom = urlparse(url).netloc
    if not allow_external and dom != current_domain:
        return False
    # Deny checks first (global/domain)
    if matches_patterns(url, denylist_global):
        return False
    if denylist_by_domain and matches_patterns(url, denylist_by_domain.get(dom, [])):
        return False
    # Allow checks (if any provided). If allowlist present, require a match.
    allow_any = (allowlist_global and len(allowlist_global) > 0) or (allowlist_by_domain and len(allowlist_by_domain.get(dom, [])) > 0)
    if allow_any:
        if matches_patterns(url, allowlist_global) or (allowlist_by_domain and matches_patterns(url, allowlist_by_domain.get(dom, []))):
            return True
        else:
            return False
    return True


def parse_robots_crawl_delay(domain: str) -> Optional[float]:
    # Fetch robots.txt and parse Crawl-delay for any UA; we ignore Disallow.
    try:
        for scheme in ("https", "http"):
            robots_url = f"{scheme}://{domain}/robots.txt"
            resp = requests.get(robots_url, timeout=10)
            if resp.status_code >= 200 and resp.status_code < 400:
                text = resp.text
                ua_section = None
                delay: Optional[float] = None
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    lower = line.lower()
                    if lower.startswith("user-agent:"):
                        ua = lower.split(":", 1)[1].strip()
                        ua_section = ua
                    elif lower.startswith("crawl-delay:"):
                        val = lower.split(":", 1)[1].strip()
                        try:
                            delay_val = float(val)
                            # Prefer UA that matches our rotator entries else *
                            delay = delay_val
                        except Exception:
                            continue
                return delay
    except Exception:
        return None
    return None


@dataclass
class CrawlResult:
    url: str
    title: Optional[str]
    text_path: Optional[str]
    status: str
    domain: str
    error: Optional[str] = None
    duration_ms: Optional[float] = None


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name[:120].strip("-") or "page"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class CrawlFrontier:
    def __init__(self, seed_domains: List[str]):
        self.heap: List[Tuple[int, int, str]] = []
        self.seen: set = set()
        self.counter = 0
        self.seed_domains = set(seed_domains)

    def push(self, url: str, depth: int):
        if url in self.seen:
            return
        self.seen.add(url)
        domain = urlparse(url).netloc
        domain_priority = 0 if domain in self.seed_domains else 1
        heapq.heappush(self.heap, (depth, domain_priority, self.counter, url))
        self.counter += 1

    def pop(self) -> Optional[Tuple[str, int]]:
        if not self.heap:
            return None
        depth, domain_priority, _, url = heapq.pop(self.heap)
        return url, depth

    def __len__(self):
        return len(self.heap)


def process_url(current_url: str, depth: int, output_dir: str, timeout: int, allow_external: bool, crawl_depth: int, ua_rotator: UserAgentRotator, rate_limiter: PerDomainRateLimiter, max_retries: int, backoff_base: float, backoff_jitter: float, allowlist_global: Optional[List[str]], denylist_global: Optional[List[str]], allowlist_by_domain: Optional[Dict[str, List[str]]], denylist_by_domain: Optional[Dict[str, List[str]]]) -> Tuple[CrawlResult, List[Tuple[str, int]]]:
    domain = urlparse(current_url).netloc
    # Adjust per-domain delay once from robots.txt if available
    if rate_limiter.get_delay(domain) == rate_limiter.default_delay:
        robots_delay = parse_robots_crawl_delay(domain)
        if robots_delay is not None:
            rate_limiter.set_delay(domain, robots_delay)
    rate_limiter.wait(domain)

    start = time.perf_counter()
    ua = ua_rotator.next()
    resp, err = fetch_with_retry(current_url, timeout=timeout, user_agent=ua, max_retries=max_retries, backoff_base=backoff_base, backoff_jitter=backoff_jitter)
    if resp is None:
        elapsed = (time.perf_counter() - start) * 1000.0
        return CrawlResult(url=current_url, title=None, text_path=None, status="fetch_error", domain=domain, error=err, duration_ms=elapsed), []

    text, title, html = detect_and_extract(resp, current_url)
    elapsed = (time.perf_counter() - start) * 1000.0
    if not text:
        result = CrawlResult(url=current_url, title=None, text_path=None, status="extraction_failed", domain=domain, duration_ms=elapsed)
        nexts: List[Tuple[str, int]] = []
    else:
        text_path = save_text(text, current_url, output_dir, title)
        result = CrawlResult(url=current_url, title=title, text_path=text_path, status="ok", domain=domain, duration_ms=elapsed)
        nexts = []

    if html and depth < crawl_depth:
        for nl in extract_links(html, current_url):
            if not should_follow(nl, domain, allow_external, allowlist_global, denylist_global, allowlist_by_domain, denylist_by_domain):
                continue
            nexts.append((nl, depth + 1))

    return result, nexts


def serialize_frontier(frontier: CrawlFrontier) -> Dict:
    return {
        "heap": frontier.heap[:],  # list of (depth, domain_priority, counter, url)
        "seen": list(frontier.seen),
        "counter": frontier.counter,
        "seed_domains": list(frontier.seed_domains),
    }

def deserialize_frontier(data: Dict) -> CrawlFrontier:
    cf = CrawlFrontier(seed_domains=data.get("seed_domains", []))
    cf.heap = data.get("heap", [])
    cf.seen = set(data.get("seen", []))
    cf.counter = int(data.get("counter", 0))
    return cf

def save_crawl_state(output_dir: str, frontier: CrawlFrontier, results: List[CrawlResult], inflight_by_domain: Dict[str, int], state_path: Optional[str] = None) -> str:
    ensure_dir(output_dir)
    path = state_path or os.path.join(output_dir, "crawl_state.json")
    state = {
        "frontier": serialize_frontier(frontier),
        "results": [asdict(r) for r in results],
        "inflight_by_domain": inflight_by_domain,
        "saved_at": time.time(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path

def load_crawl_state(state_path: str) -> Tuple[CrawlFrontier, List[CrawlResult], Dict[str, int]]:
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frontier = deserialize_frontier(data["frontier"])
    results = [CrawlResult(**r) for r in data.get("results", [])]
    inflight = data.get("inflight_by_domain", {})
    return frontier, results, inflight

def compute_domain_stats(results: List[CrawlResult]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for r in results:
        d = r.domain
        if d not in stats:
            stats[d] = {"total": 0, "ok": 0, "fetch_error": 0, "extraction_failed": 0, "error": 0, "avg_ms": 0.0, "sum_ms": 0.0}
        stats[d]["total"] += 1
        stats[d][r.status] = stats[d].get(r.status, 0) + 1
        if r.duration_ms is not None:
            stats[d]["sum_ms"] += r.duration_ms
    for d in stats:
        total = stats[d]["total"]
        stats[d]["avg_ms"] = (stats[d]["sum_ms"] / total) if total else 0.0
        del stats[d]["sum_ms"]
    return stats

def generate_html_report(output_dir: str, results: List[CrawlResult], qa: Optional[Dict] = None) -> str:
    # Fallback non-templated
    html_parts = []
    html_parts.append("<!doctype html><html><head><meta charset='utf-8'><title>Crawl Report</title>")
    html_parts.append("<style>body{font-family:Arial, sans-serif; max-width: 1000px; margin: 2rem auto; padding:0 1rem;} .ok{color: #1a7f37;} .error{color:#c62828;} .grid{display:grid; grid-template-columns: 1fr 120px 1fr; gap: 8px; align-items:center;} .card{border:1px solid #ddd; padding:12px; border-radius:8px; margin-bottom:10px;} .muted{color:#666; font-size: 0.9em;} pre{white-space: pre-wrap;}</style></head><body>")
    html_parts.append("<h1>Crawl Report</h1>")
    html_parts.append(f"<p class='muted'>Output dir: {output_dir}</p>")
    # Stats
    html_parts.append("<h2>Domain Stats</h2>")
    stats = compute_domain_stats(results)
    html_parts.append("<ul>")
    for dom, s in sorted(stats.items(), key=lambda kv: (-kv[1]["total"], kv[0])):
        html_parts.append(f"<li><strong>{dom}</strong> — total: {s['total']}, ok: {s.get('ok',0)}, errors: {s.get('fetch_error',0)+s.get('extraction_failed',0)+s.get('error',0)}, avg: {s['avg_ms']:.0f} ms</li>")
    html_parts.append("</ul>")
    # Results
    html_parts.append("<h2>Results</h2>")
    for r in results:
        status_class = "ok" if r.status == "ok" else "error" if "error" in r.status else "muted"
        title = r.title or r.url
        text_link = f"<a href='file:///{r.text_path}' target='_blank'>{r.text_path}</a>" if r.text_path else "<em>n/a</em>"
        dur = f"{r.duration_ms:.0f} ms" if r.duration_ms is not None else "n/a"
        html_parts.append(f"<div class='card'><div class='grid'><div><strong>{title}</strong><br><span class='muted'>{r.url}</span></div><div class='{status_class}'>{r.status} · {dur}</div><div>{text_link}</div></div></div>")
    if qa:
        html_parts.append("<h2>Q&A</h2>")
        html_parts.append(f"<p><strong>Question:</strong> {qa.get('question','')}</p>")
        if qa.get("summary"):
            html_parts.append("<h3>Synthesized Answer</h3>")
            html_parts.append(f"<p>{qa['summary']}</p>")
        html_parts.append("<h3>Top Passages</h3>")
        html_parts.append("<ol>")
        for hit in qa.get("hits", []):
            src = hit.get("source", "")
            passage = hit.get("passage", "").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f"<li><div class='card'><div class='muted'>Source: {src}</div><pre>{passage}</pre></div></li>")
        html_parts.append("</ol>")
    html_parts.append("</body></html>")
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return report_path

def generate_html_report_jinja(output_dir: str, results: List[CrawlResult], qa: Optional[Dict] = None, page_size: int = 50, theme: str = "light") -> List[str]:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except Exception:
        # Fallback to non-templated
        return [generate_html_report(output_dir, results, qa)]
    env = Environment(
        loader=FileSystemLoader(searchpath=os.path.join(os.getcwd(), "templates")),
        autoescape=select_autoescape(["html"])
    )
    try:
        tmpl = env.get_template("report.html")
    except Exception:
        return [generate_html_report(output_dir, results, qa)]
    stats = compute_domain_stats(results)
    pages = []
    total = len(results)
    num_pages = max(1, (total + page_size - 1) // page_size)
    for i in range(num_pages):
        start = i * page_size
        end = min(total, (i + 1) * page_size)
        page_results = results[start:end]
        out = tmpl.render(
            output_dir=output_dir,
            theme=theme,
            stats=stats,
            results=page_results,
            qa=qa if i == 0 else None,
            page=i + 1,
            num_pages=num_pages,
        )
        fname = "report_page_1.html" if num_pages == 1 else f"report_page_{i+1}.html"
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        pages.append(path)
    # Write a landing page that links to page 1
    index_path = os.path.join(output_dir, "report.html")
    with open(index_path, "w", encoding="utf-8") as f:
        if num_pages == 1:
            f.write(out)  # same content
        else:
            links = "\n".join([f"<li><a href='{os.path.basename(p)}'>Page {idx+1}</a></li>" for idx, p in enumerate(pages)])
            f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>Crawl Report</title></head><body><h1>Crawl Report</h1><ul>{links}</ul></body></html>")
    return [index_path] + pages

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
    per_domain_concurrency: int = 2,
    max_retries: int = 2,
    backoff_base: float = 0.5,
    backoff_jitter: float = 0.2,
    save_state: bool = False,
    resume: bool = False,
    state_path: Optional[str] = None,
    synthesize_question: Optional[str] = None,
    top_k_for_summary: int = 5,
    progress_callback: Optional[callable] = None,
    allowlist_patterns: Optional[List[str]] = None,
    denylist_patterns: Optional[List[str]] = None,
    allowlist_by_domain: Optional[Dict[str, List[str]]] = None,
    denylist_by_domain: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[Dict]]:
    ensure_dir(output_dir)
    discovered: List[str] = []
    if resume and (state_path or os.path.exists(os.path.join(output_dir, "crawl_state.json"))):
        spath = state_path or os.path.join(output_dir, "crawl_state.json")
        try:
            frontier, results_loaded, _ = load_crawl_state(spath)
            results: List[CrawlResult] = results_loaded
        except Exception:
            frontier = None
            results = []
    else:
        frontier = None
        results = []

    if frontier is None:
        if query:
            discovered = discover_urls_by_query(query, max_results=max_results)
        if urls:
            discovered.extend(urls)

        # Normalize and deduplicate seeds
        seeds: List[str] = []
        seed_domains: List[str] = []
        seen = set()
        for u in discovered:
            nu = normalize_url(u)
            if not nu:
                continue
            if nu not in seen:
                seeds.append(nu)
                seen.add(nu)
                seed_domains.append(urlparse(nu).netloc)

        frontier = CrawlFrontier(seed_domains=seed_domains)
        for u in seeds:
            frontier.push(u, depth=0)

    ua_rotator = UserAgentRotator(user_agents or DEFAULT_USER_AGENTS)
    rate_limiter = PerDomainRateLimiter(default_delay=per_domain_delay)
    inflight_by_domain: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures: Dict = {}
        while (len(frontier) > 0 or futures) and len(results) < max_pages_total:
            # Launch tasks respecting per-domain concurrency caps
            while len(futures) < concurrency and len(results) + len(futures) < max_pages_total and len(frontier) > 0:
                popped = frontier.pop()
                if not popped:
                    break
                current_url, depth = popped
                domain = urlparse(current_url).netloc
                if inflight_by_domain.get(domain, 0) >= per_domain_concurrency:
                    frontier.push(current_url, depth)
                    break
                inflight_by_domain[domain] = inflight_by_domain.get(domain, 0) + 1
                fut = executor.submit(process_url, current_url, depth, output_dir, timeout, allow_external, crawl_depth, ua_rotator, rate_limiter, max_retries, backoff_base, backoff_jitter)
                futures[fut] = (current_url, depth, domain)

            if futures:
                for fut in as_completed(list(futures.keys())):
                    current_url, depth, domain = futures.pop(fut)
                    inflight_by_domain[domain] = max(0, inflight_by_domain.get(domain, 1) - 1)
                    try:
                        result, nexts = fut.result()
                        results.append(result)
                        for nl, nd in nexts:
                            frontier.push(nl, nd)
                        # Emit progress event if callback provided
                        if progress_callback:
                            try:
                                progress_callback({
                                    "type": "result",
                                    "result": asdict(result),
                                    "stats": compute_domain_stats(results),
                                    "queue_size": len(frontier),
                                    "completed": len(results),
                                })
                            except Exception:
                                pass
                    except Exception as e:
                        err_res = CrawlResult(url=current_url, title=None, text_path=None, status="error", domain=domain, error=f"{type(e).__name__}: {e}")
                        results.append(err_res)
                        if progress_callback:
                            try:
                                progress_callback({
                                    "type": "error",
                                    "result": asdict(err_res),
                                    "stats": compute_domain_stats(results),
                                    "queue_size": len(frontier),
                                    "completed": len(results),
                                })
                            except Exception:
                                pass
                    break

            # Periodically save state
            if save_state and (len(results) % max(1, concurrency) == 0):
                save_crawl_state(output_dir, frontier, results, inflight_by_domain, state_path)

    # Final save
    if save_state:
        save_crawl_state(output_dir, frontier, results, inflight_by_domain, state_path)

    # Save index.json (include per-domain stats)
    index_path = os.path.join(output_dir, "index.json")
    index_data = [asdict(r) for r in results]
    stats = compute_domain_stats(results)
    index_obj = {"results": index_data, "stats": stats}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_obj, f, ensure_ascii=False, indent=2)

    report_path = None
    qa_obj = None
    if synthesize_question:
        qa_obj = ask_question(output_dir, synthesize_question, top_k=top_k_for_summary, model_name="sentence-transformers/all-MiniLM-L6-v2", synthesize=True)
        # Prefer Jinja2 templated report with pagination when available
        pages = generate_html_report_jinja(output_dir, results, qa=qa_obj, page_size=50, theme="light")
        report_path = pages[0] if pages else None
    return {"results": index_data, "stats": stats, "index_path": index_path, "report_path": report_path, "qa": qa_obj}


def discover_urls_by_query(query: str, max_results: int = 10) -> List[str]:
    urls: List[str] = []
    if DDGS is None:
        return urls
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                u = r.get("href") or r.get("url")
                if u:
                    nu = normalize_url(u)
                    if nu:
                        urls.append(nu)
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

    if not passages:
        raise RuntimeError("No passages produced from documents.")

    embeddings = model.encode(passages, normalize_embeddings=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    idx_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, idx_path)
    with open(os.path.join(output_dir, "passages.json"), "w", encoding="utf-8") as f:
        json.dump({"passages": passages, "meta": meta}, f, ensure_ascii=False)

    return {"index_path": idx_path, "passages_path": os.path.join(output_dir, "passages.json")}


def summarize_passages(passages: List[str], max_sentences: int = 5) -> str:
    # Lightweight extractive summarizer: rank sentences by term frequency across passages
    if not passages:
        return ""
    text = "\n\n".join(passages)
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text[:800]
    # Build term frequency
    def tokenize(s: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", s.lower())
    tf: Dict[str, int] = {}
    for s in sentences:
        for tok in set(tokenize(s)):
            tf[tok] = tf.get(tok, 0) + 1
    # Score sentences by sum of token frequencies
    scored = []
    for s in sentences:
        score = sum(tf.get(tok, 0) for tok in tokenize(s))
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [s for _, s in scored[:max_sentences]]
    # Preserve original order where possible
    order = {s: i for i, s in enumerate(sentences)}
    top.sort(key=lambda s: order.get(s, 0))
    return " ".join(top)

def summarize_passages_llm(passages: List[str], max_tokens: int = 512) -> Optional[str]:
    # Optional: use a small HF summarization model if available
    try:
        from transformers import pipeline
        text = "\n\n".join(passages)
        if not text.strip():
            return None
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        # Split long text to fit model input constraints
        max_chunk = 2000
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        outputs = []
        for ch in chunks:
            out = summarizer(ch, max_length=min(200, max_tokens), min_length=32, do_sample=False)
            if isinstance(out, list) and out:
                outputs.append(out[0].get("summary_text", ""))
        return "\n\n".join([o for o in outputs if o]).strip() or None
    except Exception:
        return None

def ask_question(output_dir: str, question: str, top_k: int = 5, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", synthesize: bool = True) -> Dict:
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

    extractive_answer = hits[0]["passage"] if hits else ""
    summary_extractive = summarize_passages([h["passage"] for h in hits], max_sentences=5) if (synthesize and hits) else ""
    summary_llm = summarize_passages_llm([h["passage"] for h in hits]) if synthesize else None
    summary = summary_llm or summary_extractive
    return {"question": question, "answer": extractive_answer, "summary": summary or "", "hits": hits}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Assistant: crawl the web (HTML/PDF/DOCX/EPUB) with concurrency, UA rotation, retry/backoff, prioritized frontier, build vector index, Q&A, resume, and reporting.")
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
    parser.add_argument("--per-domain-delay", type=float, default=0.5, help="Minimum delay (seconds) between requests to the same domain (overridden by robots.txt Crawl-delay if present).")
    parser.add_argument("--user-agents-file", type=str, help="Path to file containing user-agents (one per line).")
    parser.add_argument("--per-domain-concurrency", type=int, default=2, help="Max concurrent requests per domain.")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries on transient errors (429/5xx).")
    parser.add_argument("--backoff-base", type=float, default=0.5, help="Base backoff delay (seconds).")
    parser.add_argument("--backoff-jitter", type=float, default=0.2, help="Jitter added to backoff to avoid lockstep.")
    parser.add_argument("--save-state", action="store_true", help="Persist crawl state for resume.")
    parser.add_argument("--resume", action="store_true", help="Resume crawl from saved state.")
    parser.add_argument("--state-path", type=str, default=None, help="Path to save/load crawl state (defaults to output_dir/crawl_state.json).")
    parser.add_argument("--report", action="store_true", help="Generate HTML report summarizing crawl results.")
    parser.add_argument("--synthesize", action="store_true", help="Use summarizer to synthesize answers from top-k passages.")
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

    if args.report:
        # Generate report from existing index and optional question
        index_path = os.path.join(args.output_dir, "index.json")
        results: List[CrawlResult] = []
        stats: Dict[str, Dict[str, float]] = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Support legacy list-only format or new object format
                if isinstance(data, list):
                    for r in data:
                        results.append(CrawlResult(**r))
                    stats = compute_domain_stats(results)
                elif isinstance(data, dict):
                    for r in data.get("results", []):
                        results.append(CrawlResult(**r))
                    stats = data.get("stats", compute_domain_stats(results))
            except Exception:
                pass
        qa_obj = None
        if args.ask:
            qa_obj = ask_question(output_dir=args.output_dir, question=args.ask, top_k=args.top_k, synthesize=args.synthesize)
        # Prefer Jinja2 templated report (pagination/themes) if templates are present
        pages = generate_html_report_jinja(args.output_dir, results, qa=qa_obj, page_size=50, theme="light")
        report_path = pages[0] if pages else generate_html_report(args.output_dir, results, qa=qa_obj)
        print(json.dumps({"report_path": report_path, "pages": pages, "qa": qa_obj, "stats": stats}, ensure_ascii=False, indent=2))
        return

    if args.ask:
        res = ask_question(output_dir=args.output_dir, question=args.ask, top_k=args.top_k, synthesize=args.synthesize)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    if args.build_index:
        res = build_vector_index(output_dir=args.output_dir)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    state_path = args.state_path or os.path.join(args.output_dir, "crawl_state.json")
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
        per_domain_concurrency=args.per_domain_concurrency,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        backoff_jitter=args.backoff_jitter,
        save_state=args.save_state,
        resume=args.resume,
        state_path=state_path,
        synthesize_question=args.ask if args.synthesize else None,
        top_k_for_summary=args.top_k,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()