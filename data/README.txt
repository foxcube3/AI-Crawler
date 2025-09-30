AI Crawler Assistant data directory

Contents:
- index.json: sample index file in new-format (object with results and stats)
- jobs/: runtime job metadata and events (created by the server)
- reports/: generated HTML reports (created by the server)

Notes:
- The server writes crawl outputs (text files) under this directory.
- You can change the output_dir via the API or UI. By default it's "data".