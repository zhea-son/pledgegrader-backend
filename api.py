"""HTTP API for the React frontend: wraps the same code paths as src.main.

Run from the repo root:  uv run uvicorn src.api:app --reload
Interactive docs:        http://localhost:8000/docs
"""
import json
import logging
import threading
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import model, utils
from src.main import evaluate
from src.grading import grade_pledge, validate_text

logger = logging.getLogger(__name__)

# Keep in sync with the --method choices in src/main.py.
METHODS = ("rules", "climatebert", "llm")

state: dict = {"cfg": None, "arms": {}}
_arms_lock = threading.Lock()

cfg = utils.load_config()
arms = model.get_arms(cfg, ("rules", "climatebert", "llm"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Same setup as main(): logging, config and seed, done once at startup."""
    utils.setup_logging()
    cfg = utils.load_config()
    utils.set_seed(cfg["seed"])
    state["cfg"] = cfg
    get_arm("rules")  # Cheap, so warm it up; climatebert loads on first use.
    yield


app = FastAPI(title="PledgeGrader API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("FRONTEND_URL", ""),
        "https://pledgegrader.vercel.app",
        "http://localhost:5173",
        ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_arm(method: str):
    """Loads each method once and caches it, so requests don't reload models."""
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method '{method}'. Choose from {list(METHODS)}.")
    with _arms_lock:
        if method not in state["arms"]:
            logger.info("Loading method %s", method)
            state["arms"][method] = model.get_arms(state["cfg"], (method,))[method]
        return state["arms"][method]


def to_jsonable(obj):
    """Converts numpy scalars/arrays (common in model outputs) into plain Python types."""
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


class PredictIn(BaseModel):
    text: str = Field(min_length=1)
    method: str = "rules"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/methods")
def methods():
    """Methods and class labels, so the frontend can build its dropdown and legend."""
    return {"methods": list(METHODS), "labels": list(state["cfg"]["labels"].values())}


@app.post("/api/predict")
def predict(body: PredictIn):
    """Classifies one piece of text with the chosen method."""
    arm = get_arm(body.method)
    try:
        result = arm.predict(body.text)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed, see server logs.")
    return {"method": body.method, **to_jsonable(result)}


@app.get("/api/metrics/{method}")
def saved_metrics(method: str):
    """Returns metrics saved by a previous evaluation run, without recomputing."""
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method '{method}'.")
    path = Path(state["cfg"]["paths"]["predictions_dir"]) / f"{method}_metrics.json"
    if not path.exists():
        raise HTTPException(404, f"No saved metrics for '{method}'. Run the evaluation first.")
    return json.loads(path.read_text(encoding="utf8"))


class TextIn(BaseModel):
    text: str

@app.post("/api/grade")
def grade(body: TextIn):
    text = body.text.strip()

    try:
        validate_text(body.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return grade_pledge(text, arms, cfg)
    