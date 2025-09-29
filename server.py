from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
import os
import json
from typing import Optional, List, Dict

from ai_crawler import crawl, build_vector_index, ask_question, generate_html_report_jinja, CrawlResult

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
    # Load existing index.json to get results list
    index_path = os.path.join(req.output_dir, "index.json")
    results: List[CrawlResult] = []
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                results.append(CrawlResult(**r))
        except Exception:
            pass
    qa_obj = None
    if req.question:
        qa_obj = ask_question(output_dir=req.output_dir, question=req.question, top_k=req.top_k, synthesize=req.synthesize)
    pages = generate_html_report_jinja(req.output_dir, results, qa=qa_obj, page_size=req.page_size, theme=req.theme)
    return {"pages": pages, "qa": qa_obj}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))