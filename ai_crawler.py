import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from urllib.parse import urlparse, urljoin
import urllib.robotparser as robotparser

import requests
from bs4 import BeautifulSoup
import trafilatura

# Optional, used for query-based discovery without API keys
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None  # fallback if not available


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
)


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


def is_allowed_by_robots(url: str, user_agent: str = USER_AGENT) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt cannot be fetched, be conservative but allow
        return True


def fetch_html(url: str, timeout: int = 20) -> Tuple[Optional[str], Optional[str]]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # Try to guess title
        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None
        return resp.text, title
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def extract_text(html: str, url: str) -> Optional[str]:
    # Prefer trafilatura extraction
    try:
        downloaded = trafilatura.extract(html, url=url, include_comments=False)
        if downloaded and downloaded.strip():
            return downloaded.strip()
    except Exception:
        pass
    # Fallback: simple text extraction with BeautifulSoup
    try:
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        text = soup.get_text(separator="\n")
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    return None


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


def save_text(text: str, url: str, output_dir: str, title: Optional[str]) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc
    # Create a slug from title or URL path
    slug_source = title or parsed.path or "page"
    slug = sanitize_filename(slug_source)
    dir_path = os.path.join(output_dir, domain)
    ensure_dir(dir_path)
    file_path = os.path.join(dir_path, f"{slug}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path


def crawl_url(url: str, output_dir: str, timeout: int = 20) -> CrawlResult:
    parsed = urlparse(url)
    domain = parsed.netloc
    if not is_allowed_by_robots(url):
        return CrawlResult(url=url, title=None, text_path=None, status="blocked_by_robots", domain=domain)

    html, err = fetch_html(url, timeout=timeout)
    if html is None:
        return CrawlResult(url=url, title=None, text_path=None, status="fetch_error", domain=domain, error=err)

    text = extract_text(html, url)
    if not text:
        return CrawlResult(url=url, title=None, text_path=None, status="extraction_failed", domain=domain)

    # Get title again for filename; fetch_html gave either title or error string
    _, title = fetch_html(url, timeout=timeout)
    title = None if title and "Error" in title else title
    text_path = save_text(text, url, output_dir, title)
    return CrawlResult(url=url, title=title, text_path=text_path, status="ok", domain=domain)


def expand_links(html: str, base_url: str, limit: int) -> List[str]:
    if limit <= 0:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    base_domain = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme.startswith("http") and parsed.netloc == base_domain:
            links.append(abs_url)
        if len(links) >= limit:
            break
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in links:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def crawl(
    query: Optional[str],
    urls: Optional[List[str]],
    output_dir: str = "data",
    max_results: int = 10,
    max_pages_per_site: int = 1,
    timeout: int = 20,
) -> Dict[str, List[Dict]]:
    ensure_dir(output_dir)
    discovered: List[str] = []
    if query:
        discovered = discover_urls_by_query(query, max_results=max_results)
    if urls:
        discovered.extend(urls)

    # Deduplicate
    seen = set()
    seeds: List[str] = []
    for u in discovered:
        if u not in seen:
            seeds.append(u)
            seen.add(u)

    results: List[CrawlResult] = []
    for u in seeds:
        # Fetch first page
        html, err = fetch_html(u, timeout=timeout)
        if html is None:
            r = CrawlResult(url=u, title=None, text_path=None, status="fetch_error", domain=urlparse(u).netloc, error=err)
            results.append(r)
            continue

        # Crawl and save first page
        first_res = crawl_url(u, output_dir=output_dir, timeout=timeout)
        results.append(first_res)

        # Expand links within the same site (optional)
        extra_links = expand_links(html, base_url=u, limit=max_pages_per_site)
        for ex in extra_links:
            # Avoid recrawling the same URL
            if any(r.url == ex for r in results):
                continue
            ex_res = crawl_url(ex, output_dir=output_dir, timeout=timeout)
            results.append(ex_res)

        # Be polite between sites
        time.sleep(0.5)

    # Save index.json
    index_path = os.path.join(output_dir, "index.json")
    index_data = [asdict(r) for r in results]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    return {"results": index_data, "index_path": index_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Assistant: crawl the web and extract full text.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--query", type=str, help="Search query to discover pages (DuckDuckGo).")
    group.add_argument("--urls", nargs="+", help="Specific URLs to crawl.")
    parser.add_argument("--max-results", type=int, default=10, help="Max search results to crawl.")
    parser.add_argument("--max-pages-per-site", type=int, default=1, help="Extra pages to follow per site (same domain).")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for text files and index.json.")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main():
    args = parse_args()
    res = crawl(
        query=args.query,
        urls=args.urls,
        output_dir=args.output_dir,
        max_results=args.max_results,
        max_pages_per_site=args.max_pages_per_site,
        timeout=args.timeout,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()