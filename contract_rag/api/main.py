"""
main.py — FastAPI application entry point for ContractRAG.
"""

import logging
import logging.config
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contract_rag.api.routes import router
from contract_rag.retrieval.retriever import Retriever
from contract_rag.generation.generator import Generator
from contract_rag.generation.llm import OllamaClient
from contract_rag.generation.risk import RiskAnalyser
from contract_rag.generation.compare import ContractComparer
from contract_rag.generation.health import HealthScorer


# ------------------------------------------------------------------
# logging setup
# ------------------------------------------------------------------

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log", encoding="utf-8"),
        ],
    )


# ------------------------------------------------------------------
# config loader
# ------------------------------------------------------------------

def load_config(path: str = "contract_rag/configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# lifespan — startup / shutdown
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("ContractRAG starting up...")

    config = load_config()

    # initialise all components once at startup, share via app.state
    app.state.config = config
    app.state.llm = OllamaClient(config)
    app.state.retriever = Retriever(config)
    app.state.generator = Generator(config)
    app.state.risk_analyser = RiskAnalyser(config)
    app.state.comparer = ContractComparer(config)
    app.state.health_scorer = HealthScorer(config)

    logger.info("All components initialised. ContractRAG ready.")
    yield

    logger.info("ContractRAG shutting down.")


# ------------------------------------------------------------------
# app
# ------------------------------------------------------------------

app = FastAPI(
    title="ContractRAG",
    description="Contract Intelligence & Risk Review Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "contract_rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )