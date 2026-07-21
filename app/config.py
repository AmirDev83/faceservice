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

# ── Upload foto — dikirim via HTTP ke apisimak, BUKAN tulis file lokal ────────
# Sebelum Tahap 2: faceservice & apisimak wajib 1 host fisik (shared volume
# bind-mount). Setelah faceservice pindah ke server terpisah (server-utama),
# shared volume tidak jalan lintas host — foto dikirim ke endpoint internal
# apisimak (yang fisiknya satu host dengan folder uploads) lewat HTTP.
# apisimak_app cuma php-fpm (tidak bicara HTTP langsung) — WAJIB lewat
# apisimak_nginx (entrypoint HTTP-nya). Default ini untuk kondisi SEKARANG
# (1 host, network Docker sama). Setelah faceservice pindah ke server-utama,
# ganti ke URL yang bisa dijangkau lintas host, mis. http://192.168.2.2:8080/...
APISIMAK_INTERNAL_URL   = os.getenv("APISIMAK_INTERNAL_URL", "http://apisimak_nginx/api/internal/foto")
APISIMAK_INTERNAL_TOKEN = os.getenv("APISIMAK_INTERNAL_TOKEN", "")

# Dipakai HANYA oleh scripts/extract_embeddings.py (bulk Phase 2, baca file lokal
# langsung) — endpoint /verify & /extract TIDAK memakai ini lagi (lihat di atas).
# Skrip itu wajib dijalankan dari host yang sama dengan folder fisiknya (server-app).
FOTO_FACE_PATH = os.getenv("FOTO_FACE_PATH", "/var/www/uploads/foto_face")
