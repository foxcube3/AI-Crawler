AI Internet Crawler Assistant
=============================

This repository provides a simple, practical AI assistant that can search the web and crawl pages to extract full text content. It uses DuckDuckGo for discovery (no API keys required) and robust HTML text extraction with Trafilatura.

Features
- Query the web via DuckDuckGo and collect result URLs
- Respect robots.txt before crawling
- Extract clean, readable text from pages (articles, blogs, docs)
- Save outputs as .txt files and an index.json with metadata
- CLI interface with configurable limits and output directory

Setup
1) Create a virtual environment and install dependencies:
   python -m venv .venv
   . .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

Usage
- Search by query and crawl top results:
   python ai_crawler.py --query "large language models retrieval" --max-results 20 --output-dir data

- Crawl specific URLs:
   python ai_crawler.py --urls https://example.com/article https://another.com/post --output-dir data

Options
- --query: Search query to discover pages
- --urls: One or more URLs to crawl (space-separated)
- --max-results: Max number of search results to crawl (default: 10)
- --max-pages-per-site: Limit pages per domain when expanding links (default: 1; set 0 to disable)
- --output-dir: Directory to save text files and index.json (default: data)
- --timeout: HTTP timeout in seconds (default: 20)

Output
- Text files: data/{domain}/{slug}.txt
- Index file: data/index.json with metadata (URL, title, file path, status)

Notes
- Trafilatura is used for robust text extraction. If extraction fails, a simple BeautifulSoup fallback is used.
- The crawler sets a desktop User-Agent and obeys robots