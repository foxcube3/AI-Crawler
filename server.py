from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import json
import time
from typing import Optional, List, Dict

from ai_crawler import crawl, build_vector_index, ask_question, generate_html_report_jinja, CrawlResult, compute_domain_stats

# Jinja2 environment for minimal frontend
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(
        loader=FileSystemLoader(searchpath=os.path.join(os.getcwd(), "templates")),
        autoescape=select_autoescape(["html"])
    )
except Exception:
    env = None

app = FastAPI(title="AI Crawler Assistant API")

class CrawlRequest(BaseModel):
    query: Optional[str] = None
    urls: Optional[List[str]] = None
    output_dir: str = "data"
    max_results: int = 10
    crawl_depth: int = 1
    max_pages_total: int = 50
    timeout: int = 20
    allow_external: bool = True
    concurrency: int = 4
    per_domain_delay: float = 0.5
    per_domain_concurrency: int = 2
    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_jitter: float = 0.2
    save_state: bool = False
    resume: bool = False
    state_path: Optional[str] = None
    synthesize_question: Optional[str] = None
    top_k_for_summary: int = 5
    allowlist_patterns: Optional[List[str]] = None
    denylist_patterns: Optional[List[str]] = None
    allowlist_by_domain: Optional[Dict[str, List[str]]] = None
    denylist_by_domain: Optional[Dict[str, List[str]]] = None
    domain_delay_overrides: Optional[Dict[str, float]] = None

class AskRequest(BaseModel):
    output_dir: str = "data"
    question: str
    top_k: int = 5
    synthesize: bool = True

class IndexRequest(BaseModel):
    output_dir: str = "data"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

class ReportRequest(BaseModel):
    output_dir: str = "data"
    question: Optional[str] = None
    top_k: int = 5
    synthesize: bool = True
    page_size: int = 50
    theme: str = "light"

@app.post("/crawl")
def api_crawl(req: CrawlRequest):
    res = crawl(
        query=req.query,
        urls=req.urls,
        output_dir=req.output_dir,
        max_results=req.max_results,
        crawl_depth=req.crawl_depth,
        max_pages_total=req.max_pages_total,
        timeout=req.timeout,
        allow_external=req.allow_external,
        concurrency=req.concurrency,
        per_domain_delay=req.per_domain_delay,
        user_agents=None,
        per_domain_concurrency=req.per_domain_concurrency,
        max_retries=req.max_retries,
        backoff_base=req.backoff_base,
        backoff_jitter=req.backoff_jitter,
        save_state=req.save_state,
        resume=req.resume,
        state_path=req.state_path,
        synthesize_question=req.synthesize_question,
        top_k_for_summary=req.top_k_for_summary,
        progress_callback=None,
        allowlist_patterns=req.allowlist_patterns,
        denylist_patterns=req.denylist_patterns,
        allowlist_by_domain=req.allowlist_by_domain,
        denylist_by_domain=req.denylist_by_domain,
        domain_delay_overrides=req.domain_delay_overrides,
    )
    return res

@app.post("/build-index")
def api_build_index(req: IndexRequest):
    res = build_vector_index(output_dir=req.output_dir, model_name=req.model_name)
    return res

@app.post("/ask")
def api_ask(req: AskRequest):
    res = ask_question(output_dir=req.output_dir, question=req.question, top_k=req.top_k, synthesize=req.synthesize)
    return res

@app.post("/report")
def api_report(req: ReportRequest):
    # Load existing index.json to get results list and stats
    index_path = os.path.join(req.output_dir, "index.json")
    results: List[CrawlResult] = []
    stats: Dict[str, Dict] = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # legacy format
                for r in data:
                    results.append(CrawlResult(**r))
                stats = compute_domain_stats(results)
            else:
                for r in data.get("results", []):
                    results.append(CrawlResult(**r))
                stats = data.get("stats", compute_domain_stats(results))
        except Exception:
            pass
    qa_obj = None
    if req.question:
        qa_obj = ask_question(output_dir=req.output_dir, question=req.question, top_k=req.top_k, synthesize=req.synthesize)
    pages = generate_html_report_jinja(req.output_dir, results, qa=qa_obj, page_size=req.page_size, theme=req.theme)
    return {"pages": pages, "qa": qa_obj, "stats": stats}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    if env:
        try:
            tmpl = env.get_template("ui.html")
            return tmpl.render()
        except Exception:
            pass
    # Fallback HTML
    return """<!doctype html><html><head><meta charset='utf-8'><title>AI Crawler Assistant</title></head>
    <body><h1>AI Crawler Assistant</h1>
    <p>Frontend template not found. Use API endpoints: /crawl, /build-index, /ask, /report</p>
    </body></html>"""

# Serve generated report pages directly
@app.get("/reports", response_class=HTMLResponse)
def get_report(output_dir: str = "data", file: str = "report.html"):
    path = os.path.join(output_dir, file)
    if not os.path.exists(path):
        return HTMLResponse(content=f"<h1>404</h1><p>Report file not found: {path}</p>", status_code=404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{e}</p>", status_code=500)

# Serve text files for preview
@app.get("/text", response_class=HTMLResponse)
def get_text(path: str):
    if not path or not os.path.exists(path) or not path.endswith(".txt"):
        return HTMLResponse(content=f"<h1>404</h1><p>Text file not found or invalid: {path}</p>", status_code=404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Simple HTML wrapper for preview
        safe = content.replace("<", "&lt;").replace(">", "&gt;")
        return HTMLResponse(content=f"<html><body><pre>{safe}</pre></body></html>", status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{e}</p>", status_code=500)

# SSE streaming of crawl progress
try:
    from sse_starlette.sse import EventSourceResponse
except Exception:
    EventSourceResponse = None

# In-memory job store
from uuid import uuid4
from queue import Queue
import threading

JOBS: Dict[str, Dict] = {}  # job_id -> {"queue": Queue, "done": dict, "thread": Thread, "events_path": str, "cancel": dict, "output_dir": str}

@app.post("/jobs")
def create_job(req: CrawlRequest):
    # Create a job, start a crawl thread, return job_id
    job_id = str(uuid4())
    q: Queue = Queue()
    done = {"value": False}
    cancel = {"value": False}
    outdir = req.output_dir or "data"
    jobs_dir = os.path.join(outdir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    events_path = os.path.join(jobs_dir, f"{job_id}.jsonl")
    meta_path = os.path.join(jobs_dir, f"{job_id}.meta.json")
    # Persist job metadata
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "job_id": job_id,
                "created_at": time.time(),
                "params": req.dict(),
                "output_dir": outdir,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    def progress(event: Dict):
        # persist to disk
        try:
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass
        q.put(event)

    def should_stop():
        return cancel["value"]

    def run_crawl():
        try:
            crawl(
                query=req.query,
                urls=req.urls,
                output_dir=outdir,
                max_results=req.max_results,
                crawl_depth=req.crawl_depth,
                max_pages_total=req.max_pages_total,
                timeout=req.timeout,
                allow_external=req.allow_external,
                concurrency=req.concurrency,
                per_domain_delay=req.per_domain_delay,
                user_agents=None,
                per_domain_concurrency=req.per_domain_concurrency,
                max_retries=req.max_retries,
                backoff_base=req.backoff_base,
                backoff_jitter=req.backoff_jitter,
                save_state=req.save_state,
                resume=req.resume,
                state_path=req.state_path,
                synthesize_question=None,
                top_k_for_summary=req.top_k_for_summary,
                progress_callback=progress,
                allowlist_patterns=req.allowlist_patterns,
                denylist_patterns=req.denylist_patterns,
                allowlist_by_domain=req.allowlist_by_domain,
                denylist_by_domain=req.denylist_by_domain,
                should_stop=should_stop,
                domain_delay_overrides=req.domain_delay_overrides,
            )
        finally:
            done["value"] = True
            # Update metadata with completion time and generate report
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["completed_at"] = time.time()
                # Attempt to load results from index.json and generate report pages
                index_path = os.path.join(outdir, "index.json")
                results: List[CrawlResult] = []
                if os.path.exists(index_path):
                    try:
                        with open(index_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            for r in data:
                                results.append(CrawlResult(**r))
                        else:
                            for r in data.get("results", []):
                                results.append(CrawlResult(**r))
                        # Create paginated themed report (no QA by default)
                        pages = generate_html_report_jinja(outdir, results, qa=None, page_size=50, theme="light")
                        meta["report_pages"] = pages
                        meta["index_path"] = index_path
                    except Exception:
                        meta["report_pages"] = []
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            q.put({"event": "complete"})

    t = threading.Thread(target=run_crawl, daemon=True)
    t.start()
    JOBS[job_id] = {"queue": q, "done": done, "thread": t, "events_path": events_path, "cancel": cancel, "output_dir": outdir, "meta_path": meta_path}
    return {"job_id": job_id}

@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    if EventSourceResponse is None:
        return {"error": "SSE not available (install sse-starlette)."}

    job = JOBS.get(job_id)
    if not job:
        # Try to stream from persisted events if available
        # Assuming default data directory; in a real system, we'd index jobs by output_dir too.
        return {"error": f"job not found: {job_id}"}

    q: Queue = job["queue"]
    done = job["done"]
    events_path = job.get("events_path")

    async def event_generator():
        import asyncio
        # First, yield persisted events if any
        if events_path and os.path.exists(events_path):
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield {"event": "progress", "data": line.strip()}
            except Exception:
                pass
        # Then live events
        while not done["value"] or not q.empty():
            try:
                ev = q.get(timeout=0.5)
                yield {"event": "progress", "data": json.dumps(ev)}
            except Exception:
                await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())

# Legacy single-shot stream endpoint retained for convenience
@app.post("/stream-crawl")
async def stream_crawl(req: CrawlRequest):
    if EventSourceResponse is None:
        return {"error": "SSE not available (install sse-starlette)."}

    q: Queue = Queue()
    done = {"value": False}

    def progress(event: Dict):
        q.put(event)

    def run_crawl():
        try:
            crawl(
                query=req.query,
                urls=req.urls,
                output_dir=req.output_dir,
                max_results=req.max_results,
                crawl_depth=req.crawl_depth,
                max_pages_total=req.max_pages_total,
                timeout=req.timeout,
                allow_external=req.allow_external,
                concurrency=req.concurrency,
                per_domain_delay=req.per_domain_delay,
                user_agents=None,
                per_domain_concurrency=req.per_domain_concurrency,
                max_retries=req.max_retries,
                backoff_base=req.backoff_base,
                backoff_jitter=req.backoff_jitter,
                save_state=req.save_state,
                resume=req.resume,
                state_path=req.state_path,
                synthesize_question=None,
                top_k_for_summary=req.top_k_for_summary,
                progress_callback=progress,
                allowlist_patterns=req.allowlist_patterns,
                denylist_patterns=req.denylist_patterns,
                allowlist_by_domain=req.allowlist_by_domain,
                denylist_by_domain=req.denylist_by_domain,
            )
        finally:
            done["value"] = True
            q.put({"event": "complete"})

    threading.Thread(target=run_crawl, daemon=True).start()

    async def event_generator():
        import asyncio
        while not done["value"] or not q.empty():
            try:
                ev = q.get(timeout=0.5)
                yield {"event": "progress", "data": json.dumps(ev)}
            except Exception:
                await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())

# Job status and cancellation
@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        # Try to load metadata only
        # Search in default data dir
        return {"error": f"job not found: {job_id}"}
    done = job["done"]["value"]
    # Read last event if exists
    last_event = None
    events_path = job.get("events_path")
    if events_path and os.path.exists(events_path):
        try:
            with open(events_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                chunk = 1024
                data = b""
                while size > 0:
                    size = max(0, size - chunk)
                    f.seek(size)
                    data = f.read(chunk) + data
                    if b"\n" in data:
                        break
                lines = data.split(b"\n")
                for line in reversed(lines):
                    if line.strip():
                        last_event = json.loads(line.decode("utf-8"))
                        break
        except Exception:
            pass
    meta = {}
    meta_path = job.get("meta_path")
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    return {"done": done, "last_event": last_event, "meta": meta}

@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": f"job not found: {job_id}"}
    job["cancel"]["value"] = True
    return {"job_id": job_id, "cancelled": True}

@app.get("/jobs")
def list_jobs(output_dir: str = "data"):
    jobs_dir = os.path.join(output_dir, "jobs")
    if not os.path.exists(jobs_dir):
        return {"jobs": []}
    jobs = []
    for fn in os.listdir(jobs_dir):
        if fn.endswith(".meta.json"):
            path = os.path.join(jobs_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                jid = meta.get("job_id") or fn.split(".")[0]
                # Determine status
                active = jid in JOBS
                done = False
                if active:
                    done = JOBS[jid]["done"]["value"]
                else:
                    done = bool(meta.get("completed_at"))
                jobs.append({"job_id": jid, "active": active, "done": done, "meta": meta})
            except Exception:
                continue
    return {"jobs": jobs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))