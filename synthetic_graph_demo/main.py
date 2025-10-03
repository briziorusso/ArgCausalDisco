import asyncio

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from utils import (
    generate_causal_graph,
    get_bifxml,
    heuristic_by_degrees,
    heuristic_by_semantics,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MAX_NODES = 20
HEURISTICS = {
    "degrees": heuristic_by_degrees,
    "semantics": heuristic_by_semantics,
    "none": lambda _, iterable: next(iterable, None),
}


class DotRequest(BaseModel):
    dot: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/generate_dag", response_class=PlainTextResponse)
async def generate_dag(
    n_nodes: int = 6,
    n_edges: int = 6,
    quality: int = 3,
    heuristic: str = "degrees",
    graph_type: str = "random",
    w_compact: float | None = None,
    w_specificity: float | None = None,
    w_correlation: float | None = None,
    excluded_concepts: str | None = None,
):
    """
    Generates a random Directed Acyclic Graph (DAG) and returns its DOT representation.
    """
    n_nodes = max(min(n_nodes, MAX_NODES), 4)
    max_edges = int(n_nodes * (n_nodes - 1) / 2)
    n_edges = max(min(n_edges, max_edges), n_nodes - 1)
    quality = max(min(quality, 10), 1)

    # Log scale for candidates
    candidates_log = np.logspace(np.log10(10), np.log10(1000), 10, dtype=int)
    subgraph_candidates = candidates_log[quality - 1]

    try:
        # Run the graph generation function directly in an async context
        heuristic_func = HEURISTICS[heuristic]
        if heuristic == "semantics" and any(
            v is not None for v in (w_compact, w_specificity, w_correlation)
        ):
            # Bind weights partially when provided
            def semantic_with_weights(cause_net, iterable):
                return heuristic_by_semantics(
                    cause_net,
                    iterable,
                    w_compact=w_compact if w_compact is not None else 0,
                    w_specificity=w_specificity if w_specificity is not None else 0,
                    w_correlation=w_correlation if w_correlation is not None else 1,
                )

            heuristic_func = semantic_with_weights

        excluded_concepts_set = (
            set(excluded_concepts.split(",")) if excluded_concepts else None
        )
        return await asyncio.to_thread(
            generate_causal_graph,
            n_nodes,
            n_edges,
            subgraph_candidates,
            heuristic_func,
            graph_type,
            excluded_concepts=excluded_concepts_set,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/convert")
async def proxy_convert(request: DotRequest):
    try:
        bifxml_content = get_bifxml(request.dot)
        return {"bifxml": bifxml_content}
    except Exception as e:
        return {"error": str(e)}
