from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import json
import time
import threading
import sqlite3
from typing import Optional, List, Dict

from ai_crawler import crawl, build_vector_index, ask_question, generate_html_report_jinja, CrawlResult, compute_domain_stats

# Optional cron scheduling
try:
    from croniter import croniter
except Exception:
    croniter = None

DB_PATH = os.path.join("data", "app.db")

def _init_db():
    try:
        os.makedirs("data", exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS schedules (schedule_id TEXT PRIMARY KEY, interval_seconds INTEGER, cron_expr TEXT, params_json TEXT, created_at REAL, active INTEGER)")
        cur.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, output_dir TEXT, created_at REAL, completed_at REAL, status TEXT)")
        # Persist crawl events for analytics
        cur.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts REAL, type TEXT, data_json TEXT)")
        conn.commit()
    finally:
        conn.close()

def _db_execute(query: str, args: tuple = ()):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(query, args)
        conn.commit()
    finally:
        conn.close()

def _db_query(query: str, args: tuple = ()) -> List[tuple]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(query, args)
        return cur.fetchall()
    finally:
        conn.close()

# Schedules persistence helpers
def _schedules_file(output_dir: Optional[str] = None) -> str:
    out = output_dir or "data"
    try:
        os.makedirs(out, exist_ok=True)
    except Exception:
        pass
    return os.path.join(out, "schedules.json")

def _serialize_schedule(schedule_id: str, sched: Dict) -> Dict:
    return {
        "schedule_id": schedule_id,
        "interval_seconds": sched.get("interval", 0),
        "params": sched.get("params").dict() if hasattr(sched.get("params"), "dict") else sched.get("params"),
        "output_dir": (sched.get("params").output_dir if hasattr(sched.get("params"), "output_dir") else sched.get("output_dir", "data")),
        "created_at": sched.get("created_at", time.time()),
        "active": True,
    }

def _save_schedules():
    # Persist all schedules to the default "data" dir and to each schedule's output_dir for redundancy
    items = []
    for sid, s in SCHEDULES.items():
        try:
            items.append(_serialize_schedule(sid, s))
        except Exception:
            continue
    # Default write
    try:
        with open(_schedules_file("data"), "w", encoding="utf-8") as f:
            json.dump({"schedules": items}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # Per-output-dir write
    by_outdir: Dict[str, List[Dict]] = {}
    for it in items:
        od = it.get("output_dir") or "data"
        by_outdir.setdefault(od, []).append(it)
    for od, lst in by_outdir.items():
        try:
            with open(_schedules_file(od), "w", encoding="utf-8") as f:
                json.dump({"schedules": lst}, f, ensure_ascii=False, indent=2)
        except Exception:
            continue

def _load_schedules_from_disk():
    # Load schedules from default "data/schedules.json"
    path = _schedules_file("data")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in (data.get("schedules") or []):
            try:
                sid = it.get("schedule_id") or str(time.time())
                if sid in SCHEDULES:
                    continue
                # Recreate CrawlRequest from dict
                params_dict = it.get("params") or {}
                # Minimal recreation; validation handled by Pydantic
                params = CrawlRequest(**params_dict)
                cancel = {"value": False}
                SCHEDULES[sid] = {"interval": int(it.get("interval_seconds", 0)), "params": params, "cancel": cancel, "created_at": it.get("created_at", time.time())}
                t = threading.Thread(target=_schedule_loop, args=(sid,), daemon=True)
                SCHEDULES[sid]["thread"] = t
                t.start()
            except Exception:
                continue
    except Exception:
        pass

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

@app.get("/jobs/{job_id}/detail", response_class=HTMLResponse)
def job_detail_page(job_id: str):
    if env:
        try:
            tmpl = env.get_template("job_detail.html")
            return tmpl.render(job_id=job_id)
        except Exception:
            pass
    return HTMLResponse(f"<h1>Job {job_id}</h1><p>Template not found. Use /jobs/{job_id} for JSON.</p>", status_code=200)

@app.get("/validate-cron")
def validate_cron(expr: str):
    if not expr:
        return {"valid": False, "error": "empty"}
    if croniter is None:
        return {"valid": False, "error": "croniter not installed"}
    try:
        base = time.time()
        it = croniter(expr, base)
        next_run = it.get_next(float)
        return {"valid": True, "next_run_epoch": next_run}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/analytics/domain-stats")
def analytics_domain_stats(job_id: Optional[str] = None):
    # Aggregate domain stats from DB events (type='result')
    where = ""
    args: List = []
    if job_id:
        where = "WHERE job_id=?"
        args = [job_id]
    rows = _db_query(f"SELECT data_json FROM events {where}", tuple(args))
    results: List[CrawlResult] = []
    for (dj,) in rows:
        try:
            ev = json.loads(dj)
            res = ev.get("result")
            if isinstance(res, dict) and res.get("url"):
                results.append(CrawlResult(**res))
        except Exception:
            continue
    stats = compute_domain_stats(results) if results else {}
    return {"count": len(results), "stats": stats}

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
    now = time.time()
    # Persist job metadata (file)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "job_id": job_id,
                "created_at": now,
                "params": req.dict(),
                "output_dir": outdir,
                "status": "running",
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # Persist job in DB
    try:
        _db_execute("INSERT OR REPLACE INTO jobs(job_id, output_dir, created_at, completed_at, status) VALUES(?,?,?,?,?)",
                    (job_id, outdir, now, None, "running"))
    except Exception:
        pass

    def progress(event: Dict):
        # persist to disk
        try:
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass
        # persist to DB for analytics
        try:
            _db_execute("INSERT INTO events(job_id, ts, type, data_json) VALUES(?,?,?,?)", (job_id, time.time(), event.get("type") or "result", json.dumps(event)))
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
                meta["status"] = "done" if not cancel["value"] else "cancelled"
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
            # Update DB
            try:
                _db_execute("UPDATE jobs SET completed_at=?, status=? WHERE job_id=?", (time.time(), "done" if not cancel["value"] else "cancelled", job_id))
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
    done = False
    meta = {}
    last_event = None
    events_path = None
    if job:
        done = job["done"]["value"]
        events_path = job.get("events_path")
        meta_path = job.get("meta_path")
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
    else:
        # Try loading persisted meta and events when job is not active
        outdir = "data"
        jobs_dir = os.path.join(outdir, "jobs")
        meta_path = os.path.join(jobs_dir, f"{job_id}.meta.json")
        events_path = os.path.join(jobs_dir, f"{job_id}.jsonl")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        done = bool(meta.get("completed_at"))

    # Read last event if exists
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

@app.get("/jobs/{job_id}")
def job_detail(job_id: str, output_dir: str = "data"):
    # Return meta, aggregated stats (from events), and report links
    # Try active job store first
    job = JOBS.get(job_id)
    meta = {}
    events_path = None
    if job:
        meta_path = job.get("meta_path")
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        events_path = job.get("events_path")
    else:
        # Look up persisted meta
        jobs_dir = os.path.join(output_dir, "jobs")
        mpath = os.path.join(jobs_dir, f"{job_id}.meta.json")
        if os.path.exists(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        events_path = os.path.join(jobs_dir, f"{job_id}.jsonl")

    # Aggregate events into CrawlResult list and compute stats
    results: List[CrawlResult] = []
    events_count = 0
    if events_path and os.path.exists(events_path):
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    events_count += 1
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    res = ev.get("result")
                    if isinstance(res, dict) and res.get("url"):
                        try:
                            results.append(CrawlResult(**res))
                        except Exception:
                            continue
        except Exception:
            pass
    stats = compute_domain_stats(results) if results else {}
    return {"job_id": job_id, "meta": meta, "events_count": events_count, "stats": stats}

# ---------- Simple in-memory scheduler ----------

SCHEDULES: Dict[str, Dict] = {}  # schedule_id -> {"interval": int, "params": dict, "cancel": dict, "thread": Thread, "output_dir": str}

class ScheduleRequest(BaseModel):
    interval_seconds: Optional[int] = None
    cron_expr: Optional[str] = None
    params: CrawlRequest

def _schedule_loop(schedule_id: str):
    sched = SCHEDULES.get(schedule_id)
    if not sched:
        return
    cancel = sched["cancel"]
    params: CrawlRequest = sched["params"]
    interval = sched.get("interval")
    cron = sched.get("cron_expr")
    if cron and croniter:
        base = time.time()
        it = croniter(cron, base)
        next_run = it.get_next(float)
        while not cancel["value"]:
            now = time.time()
            if now >= next_run:
                try:
                    create_job(params)
                except Exception:
                    pass
                next_run = it.get_next(float)
            time.sleep(0.5)
    else:
        interval = max(1, int(interval or 60))
        while not cancel["value"]:
            try:
                create_job(params)
            except Exception:
                pass
            for _ in range(interval):
                if cancel["value"]:
                    break
                time.sleep(1)

@app.post("/schedules")
def create_schedule(req: ScheduleRequest):
    schedule_id = str(uuid4())
    cancel = {"value": False}
    SCHEDULES[schedule_id] = {
        "interval": req.interval_seconds,
        "cron_expr": req.cron_expr,
        "params": req.params,
        "cancel": cancel,
        "created_at": time.time(),
    }
    # Persist to DB
    try:
        _db_execute(
            "INSERT OR REPLACE INTO schedules(schedule_id, interval_seconds, cron_expr, params_json, created_at, active) VALUES(?,?,?,?,?,1)",
            (schedule_id, req.interval_seconds or 0, req.cron_expr or "", json.dumps(req.params.dict()), time.time()),
        )
    except Exception:
        pass
    t = threading.Thread(target=_schedule_loop, args=(schedule_id,), daemon=True)
    SCHEDULES[schedule_id]["thread"] = t
    t.start()
    _save_schedules()
    return {"schedule_id": schedule_id}

@app.get("/schedules")
def list_schedules(output_dir: str = "data"):
    # Combine in-memory schedules with persisted JSON and DB
    items = []
    for sid, s in SCHEDULES.items():
        items.append({"schedule_id": sid, "interval_seconds": s.get("interval"), "cron_expr": s.get("cron_expr"), "params": s["params"].dict(), "created_at": s.get("created_at", time.time())})
    # Read persisted file (optional)
    try:
        with open(_schedules_file(output_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in (data.get("schedules") or []):
            if not any(x["schedule_id"] == it.get("schedule_id") for x in items):
                items.append(it)
    except Exception:
        pass
    # Read from DB
    try:
        rows = _db_query("SELECT schedule_id, interval_seconds, cron_expr, params_json, created_at FROM schedules WHERE active=1")
        for sid, interval, cron, params_json, created_at in rows:
            if not any(x["schedule_id"] == sid for x in items):
                try:
                    params = json.loads(params_json)
                except Exception:
                    params = {}
                items.append({"schedule_id": sid, "interval_seconds": interval, "cron_expr": cron, "params": params, "created_at": created_at})
        # Optionally, update in-memory from DB
    except Exception:
        pass
    return {"schedules": items}

@app.delete("/schedules/{schedule_id}")
def delete_schedule(schedule_id: str):
    s = SCHEDULES.get(schedule_id)
    if not s:
        return {"error": f"schedule not found: {schedule_id}"}
    s["cancel"]["value"] = True
    # Remove from in-memory store and persist update
    try:
        del SCHEDULES[schedule_id]
    except Exception:
        pass
    _save_schedules()
    return {"schedule_id": schedule_id, "cancelled": True}

# Load schedules on startup (best-effort)
try:
    _init_db()
    # Load from DB first
    rows = _db_query("SELECT schedule_id, interval_seconds, cron_expr, params_json, created_at FROM schedules WHERE active=1")
    for sid, interval, cron, params_json, created_at in rows:
        if sid in SCHEDULES:
            continue
        try:
            params = CrawlRequest(**json.loads(params_json))
            cancel = {"value": False}
            SCHEDULES[sid] = {"interval": interval, "cron_expr": cron, "params": params, "cancel": cancel, "created_at": created_at}
            t = threading.Thread(target=_schedule_loop, args=(sid,), daemon=True)
            SCHEDULES[sid]["thread"] = t
            t.start()
        except Exception:
            continue
    # Then JSON fallback
    _load_schedules_from_disk()
except Exception:
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))