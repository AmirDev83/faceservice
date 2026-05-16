"""
ArcFace verification service menggunakan InsightFace buffalo_l.

══════════════════════════════════════════════════════════════════
ALUR TEKNIS LENGKAP
══════════════════════════════════════════════════════════════════

[DATA AWAL] tbl_siabdi_face_embeddings (db_simpeg)
  - Diisi oleh Flutter saat registrasi wajah
  - Kolom lama  : embedding_lurus/kiri/kanan/senyum/frontal  ← FaceNet on-device (Flutter)
  - Kolom foto  : foto_wajah  ← filename saja, file ada di uploads/foto_face/
  - Kolom baru  : embedding_arcface  ← diisi oleh Python (Phase 2)

[PHASE 2] scripts/extract_embeddings.py
  - Baca semua baris yang punya foto_wajah
  - Buka file gambar: FOTO_FACE_PATH/{foto_wajah}
  - Jalankan InsightFace buffalo_l → ekstrak 512-dim ArcFace embedding
  - Simpan ke embedding_arcface (JSON array)
  - Setelah selesai: semua pegawai yang punya foto master sudah punya ArcFace embedding

[PHASE 3] Verifikasi real-time (saat Flutter presensi)
  1. Flutter ambil foto selfie → POST ke Laravel /face/verify-live (foto + nip)
  2. Laravel terima foto → forward ke Python POST /verify (foto bytes + nip)
  3. Python:
     a. Decode foto → cv2 → InsightFace → probe_embedding (512-dim)
     b. Query embedding_arcface WHERE nip = ? dari DB
     c. Cosine similarity: probe vs referensi
     d. Kembalikan {verified: bool, score: float}
  4. Laravel terima hasil → kirim ke Flutter {use_device: false, verified, score}
  5. Flutter pakai hasil server (bukan FaceNet on-device)

[KEUNGGULAN vs FaceNet on-device]
  - FaceNet on-device: ~70-90% akurasi, rentan pencahayaan, 512-dim MobileNet
  - ArcFace buffalo_l : ~99.8% LFW, ResNet100, tahan pose & cahaya

[THRESHOLD]
  - Cosine similarity (dot product vektor ternormalisasi): 0.0 = beda, 1.0 = identik
  - ARCFACE_THRESHOLD=0.40 → cukup ketat, turunkan ke 0.35 jika terlalu banyak gagal
══════════════════════════════════════════════════════════════════
"""

import json
import logging
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

from app import config
from app.database import get_connection

logger = logging.getLogger(__name__)

# ── Singleton model ────────────────────────────────────────────────────────────

_face_app: FaceAnalysis | None = None


def init_model():
    global _face_app
    if _face_app is not None:
        return
    logger.info("Loading InsightFace buffalo_l model...")
    _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace buffalo_l model loaded.")


def _get_app() -> FaceAnalysis:
    if _face_app is None:
        raise RuntimeError("Model belum diinisialisasi. Panggil init_model() saat startup.")
    return _face_app


# ── Core functions ─────────────────────────────────────────────────────────────

def extract_embedding(image_bytes: bytes) -> np.ndarray | None:
    """
    Ekstrak 512-dim ArcFace embedding dari bytes JPEG/PNG.
    Return None jika wajah tidak terdeteksi.

    Berbeda dengan Flutter FaceNet:
    - Flutter: crop wajah manual → MobileNet → 512-dim
    - Python : InsightFace deteksi otomatis → ResNet100 → 512-dim L2-normalized
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.warning("Gagal decode gambar.")
        return None

    app = _get_app()
    faces = app.get(img_bgr)

    if not faces:
        logger.info("Tidak ada wajah terdeteksi.")
        return None

    # Ambil wajah terbesar (paling dekat kamera)
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.normed_embedding  # L2-normalized, shape (512,), dtype float32
    return emb.astype(np.float32)


def extract_embedding_from_file(filepath: str) -> np.ndarray | None:
    """Baca file gambar dari path → extract embedding. Untuk Phase 2 script."""
    try:
        with open(filepath, "rb") as f:
            return extract_embedding(f.read())
    except OSError as e:
        logger.error(f"Gagal buka file {filepath}: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity antara dua vektor (dot product setelah normalisasi)."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def verify_against_stored(nip: str, probe_embedding: np.ndarray) -> dict:
    """
    Bandingkan probe_embedding (dari foto selfie saat presensi) dengan
    embedding_arcface yang tersimpan di DB (diekstrak dari foto_wajah master).

    Return:
      {verified: bool, score: float}               — sukses
      {verified: False, score: 0.0, detail: str}   — gagal (tidak ada embedding)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding_arcface FROM tbl_siabdi_face_embeddings WHERE nip = %s LIMIT 1",
                (nip,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or not row.get("embedding_arcface"):
        logger.warning(f"[{nip}] embedding_arcface belum tersedia. Jalankan Phase 2 script terlebih dahulu.")
        return {"verified": False, "score": 0.0, "detail": "no_embedding"}

    ref = np.array(json.loads(row["embedding_arcface"]), dtype=np.float32)
    score = cosine_similarity(probe_embedding, ref)
    verified = score >= config.ARCFACE_THRESHOLD

    logger.info(f"[{nip}] score={score:.4f} threshold={config.ARCFACE_THRESHOLD} verified={verified}")
    return {"verified": verified, "score": round(score, 4)}


def save_embedding(nip: str, embedding: np.ndarray) -> bool:
    """
    Simpan ArcFace embedding ke kolom embedding_arcface.
    Dipanggil dari Phase 2 script dan endpoint /extract.
    """
    emb_json = json.dumps(embedding.tolist())
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tbl_siabdi_face_embeddings
                SET embedding_arcface = %s, updated_at = NOW()
                WHERE nip = %s
                """,
                (emb_json, nip),
            )
            affected = cur.rowcount
    finally:
        conn.close()

    if affected == 0:
        logger.warning(f"[{nip}] UPDATE 0 baris — NIP tidak ada di tabel.")
    return affected > 0
