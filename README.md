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
- The crawler sets a desktop User-Agent and obeys robots.txt.

Windows Installer (PyInstaller + Inno Setup)
Prerequisites (Windows 10/11)
- Python 3.10+ installed and on PATH
- Git (optional, for cloning)
- Inno Setup 6: https://jrsoftware.org/isdl.php

Step 1: Build the Windows executable
- Open Command Prompt in the project root
- Run:
  scripts\build_windows.bat
- On success, the server executable will be at:
  dist\AI_Crawler_Assistant_Server.exe

Step 2: Create the installer with Inno Setup
- Install Inno Setup 6
- Open the script:
  installers\inno_setup\installer.iss
- Compile (Build -> Compile). The installer will be generated next to the script as:
  installers\inno_setup\AI_Crawler_Assistant_Installer.exe

What the installer does
- Installs to: C:\Program Files\AI Crawler Assistant (by default)
- Bundles templates\ and a sample data\ structure
- Creates Start Menu and Desktop shortcuts for “AI Crawler Assistant Server”
- Offers to start the server after installation

Run after installation
- Use the Start Menu or Desktop shortcut “AI Crawler Assistant Server”
- The server executable is:
  C:\Program Files\AI Crawler Assistant\AI_Crawler_Assistant_Server.exe

Uninstall
- Use “Add or Remove Programs” in Windows Settings, or the Start Menu shortcut

Troubleshooting
- If Inno Setup can’t find the executable, run Step 1 to produce dist\AI_Crawler_Assistant_Server.exe
- If PyInstaller build fails, ensure templates\ exists and rerun scripts\build_windows.bat
- To include additional assets, adjust the --add-data flag in scripts\build_windows.bat and corresponding [Files] entries in installers\inno_setup\installer.iss