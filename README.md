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

Run the Server (Linux/macOS/Windows)
- From source (dev mode):
  python server.py
- Environment:
  PORT=8000 by default. Change with:
  PORT=8080 python server.py
  On Windows (CMD):
  set PORT=8080 && python server.py

Access the Web UI and API
- Web UI:
  http://localhost:8000/ui
- Jobs dashboard:
  http://localhost:8000/jobs-ui
- Report pages (after a crawl/report is generated):
  http://localhost:8000/reports?output_dir=data&file=report.html
- FastAPI interactive docs:
  http://localhost:8000/docs

Typical API Workflow
1) Crawl by query
   POST /crawl
   JSON body:
   {
     "query": "large language models retrieval",
     "output_dir": "data",
     "max_results": 10,
     "crawl_depth": 1
   }
2) Build an embedding index
   POST /build-index
   JSON body:
   {
     "output_dir": "data",
     "model_name": "sentence-transformers/all-MiniLM-L6-v2"
   }
3) Ask questions over crawled corpus (RAG)
   POST /ask
   JSON body:
   {
     "output_dir": "data",
     "question": "What are best practices for web crawling?",
     "top_k": 5,
     "synthesize": true
   }
4) Generate an HTML report
   POST /report
   JSON body:
   {
     "output_dir": "data",
     "question": "Summarize main findings",
     "top_k": 5,
     "synthesize": true,
     "page_size": 50,
     "theme": "light"
   }

Jobs API (streaming and persistence)
- Create a long-running crawl job:
  POST /jobs
  Body: same as CrawlRequest
  Response: { "job_id": "<id>" }
- Stream progress (Server-Sent Events):
  GET /jobs/{job_id}/stream
- Check status:
  GET /jobs/{job_id}/status
- List jobs:
  GET /jobs
- Cancel job:
  DELETE /jobs/{job_id}
- CSV export of jobs:
  GET /jobs-csv
Notes:
- Streaming requires sse-starlette (already in requirements.txt).
- Job metadata and events are persisted under data/jobs and data/app.db (SQLite).

Scheduling
- Create a schedule to run periodically:
  POST /schedules
  JSON body:
  {
    "interval_seconds": 3600,
    "params": { "...": "CrawlRequest fields" }
  }
- Cron expression (requires croniter):
  {
    "cron_expr": "*/15 * * * *",
    "params": { "...": "CrawlRequest fields" }
  }
- List schedules:
  GET /schedules
- Delete schedule:
  DELETE /schedules/{schedule_id}
Persistence:
- Schedules are stored in SQLite (data/app.db) and mirrored to data/schedules.json.

Data Layout
- Text content: data/{domain}/{slug}.txt
- Index file: data/index.json (may be list in legacy, or {results, stats} object)
- Jobs:
  data/jobs/{job_id}.meta.json
  data/jobs/{job_id}.jsonl (progress events)
- Reports: written to output_dir as paginated HTML files (e.g., report_1.html, report_2.html)

Customization
- Templates used by the server are under templates/
- When built with PyInstaller, templates are bundled and loaded from the executable’s resources
- You can add or modify templates (ui.html, jobs.html, job_detail.html) to change the UI

Security and Etiquette
- Respect robots.txt; crawler attempts to be polite with delays and concurrency limits
- Use allowlist/denylist patterns to control domains and paths
- Consider setting per_domain_delay and per_domain_concurrency conservatively for production crawls

Curl Examples (API)
Note: Replace localhost:8000 with your host/port.

- Crawl by query:
  curl -X POST http://localhost:8000/crawl ^
       -H "Content-Type: application/json" ^
       -d "{\"query\":\"large language models retrieval\",\"output_dir\":\"data\",\"max_results\":5,\"crawl_depth\":1}"

- Build index:
  curl -X POST http://localhost:8000/build-index ^
       -H "Content-Type: application/json" ^
       -d "{\"output_dir\":\"data\",\"model_name\":\"sentence-transformers/all-MiniLM-L6-v2\"}"

- Ask question:
  curl -X POST http://localhost:8000/ask ^
       -H "Content-Type: application/json" ^
       -d "{\"output_dir\":\"data\",\"question\":\"What are best practices for web crawling?\",\"top_k\":5,\"synthesize\":true}"

- Generate report:
  curl -X POST http://localhost:8000/report ^
       -H "Content-Type: application/json" ^
       -d "{\"output_dir\":\"data\",\"question\":\"Summarize main findings\",\"top_k\":5,\"synthesize\":true,\"page_size\":50,\"theme\":\"light\"}"

- Create job:
  curl -X POST http://localhost:8000/jobs ^
       -H "Content-Type: application/json" ^
       -d "{\"query\":\"large language models\",\"output_dir\":\"data\",\"max_results\":10}"

- Stream job progress (SSE):
  curl -N http://localhost:8000/jobs/<job_id>/stream

- Job status:
  curl http://localhost:8000/jobs/<job_id>/status

- List jobs:
  curl http://localhost:8000/jobs

- Cancel job:
  curl -X DELETE http://localhost:8000/jobs/<job_id>

- Jobs CSV export:
  curl -L "http://localhost:8000/jobs-csv?status=all"

- Validate cron (requires croniter):
  curl "http://localhost:8000/validate-cron?expr=*/15%20*%20*%20*%20*"

- Analytics domain stats:
  curl "http://localhost:8000/analytics/domain-stats"

- Serve a generated report page:
  curl "http://localhost:8000/reports?output_dir=data&file=report.html"

Quickstart PowerShell (Windows)
- Script: scripts\\quickstart.ps1
- Download/Run:
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  .\\scripts\\quickstart.ps1
- This script creates a venv, installs requirements, starts the server, opens the Web UI, and runs sample API calls.
Download: [scripts\\quickstart.ps1](file:///scripts/quickstart.ps1)

Run as a Windows Service
Option A: Using the PyInstaller executable and NSSM (recommended)
- Build the executable via scripts\\build_windows.bat (dist\\AI_Crawler_Assistant_Server.exe)
- Install NSSM: https://nssm.cc/download
- Create the service:
  nssm install AI-Crawler-Assistant "C:\\Program Files\\AI Crawler Assistant\\AI_Crawler_Assistant_Server.exe"
- Set Startup directory to the install folder and ensure it has write access to data\\
- Start the service:
  nssm start AI-Crawler-Assistant
Notes:
- Use the Inno Setup installer for easy deployment to C:\\Program Files\\AI Crawler Assistant.
- Logs can be redirected via NSSM I/O settings.

Option B: Using pythonw.exe
- Service binPath (example):
  sc create AI-Crawler-Assistant binPath= "\"C:\\Python311\\pythonw.exe\" \"C:\\path\\to\\server.py\"" start= auto
- Set the working directory in a wrapper script or run via NSSM if you hit permission issues.

Run as a systemd Service (Linux)
- Create unit file: /etc/systemd/system/ai-crawler.service
- Example:
  [Unit]
  Description=AI Crawler Assistant
  After=network.target

  [Service]
  Type=simple
  WorkingDirectory=/opt/ai-crawler
  Environment=PORT=8000
  ExecStart=/usr/bin/python3 /opt/ai-crawler/server.py
  Restart=on-failure
  User=www-data
  Group=www-data

  [Install]
  WantedBy=multi-user.target
- Commands:
  sudo systemctl daemon-reload
  sudo systemctl enable ai-crawler
  sudo systemctl start ai-crawler
- Logs:
  journalctl -u ai-crawler -f

Screenshots
- Web UI (/ui):
  docs/screenshots/ui.png
- Jobs dashboard (/jobs-ui):
  docs/screenshots/jobs-ui.png
To include screenshots:
- Create the folder docs/screenshots/
- Add ui.png and jobs-ui.png (PNG or JPG)
- The README will automatically link them if present.

Support
- Issues and feature requests: open a ticket with logs from data/jobs and console output
- Include Python version, OS, and steps to reproduce