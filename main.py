from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from routers import analyze, health

# ── Model pre-loading ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🔄 Loading AI models...")
    from services.whisper_service  import load_model as load_whisper
    from services.nlp_service      import load_nlp
    from services.eye_contact_service import load_face_mesh
    load_whisper()
    load_nlp()
    load_face_mesh()
    print("✅ All models loaded — engine ready\n")
    yield
    print("Engine shutting down.")

app = FastAPI(title="Soft Skill Analyzer Engine", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(health.router)
