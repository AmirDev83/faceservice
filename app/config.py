import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ─────────────────────────────────────────────────────────────────
# Dibaca dari environment (docker-compose inject dari apisimak/.env)
# DB_NAME  ← docker-compose memetakan ${DB_DATABASE} ke DB_NAME
DB_HOST     = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT     = int(os.getenv("DB_PORT", 3306))
DB_NAME     = os.getenv("DB_NAME", "db_simpeg")
DB_USER     = os.getenv("DB_USER", "adminsimak")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ── Service ──────────────────────────────────────────────────────────────────
FACE_SERVICE_HOST = os.getenv("FACE_SERVICE_HOST", "0.0.0.0")
FACE_SERVICE_PORT = int(os.getenv("FACE_SERVICE_PORT", 8001))

# ── ArcFace ──────────────────────────────────────────────────────────────────
ARCFACE_THRESHOLD = float(os.getenv("ARCFACE_THRESHOLD", 0.40))

# ── Path upload (Docker volume: ../simak/uploads:/var/www/uploads) ───────────
# Sama persis dengan base_path('uploads/...') di container apisimak_app
FOTO_FACE_PATH    = os.getenv("FOTO_FACE_PATH",    "/var/www/uploads/foto_face")
AUDIT_SELFIE_PATH = os.getenv("AUDIT_SELFIE_PATH", "/var/www/uploads/audit_selfie")
