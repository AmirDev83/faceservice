"""
Phase 2 — Bulk extract ArcFace embeddings dari foto_wajah yang sudah ada di DB.

Cara pakai (jalankan dari root faceservice/):
    python scripts/extract_embeddings.py

Output:
    - OK  : embedding berhasil disimpan ke embedding_arcface
    - SKIP: embedding_arcface sudah ada (--force untuk overwrite)
    - FAIL: wajah tidak terdeteksi / foto tidak ada
"""

import os
import sys
import argparse
import logging

# Tambahkan root faceservice ke sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.database import get_connection
from app.services import arcface as arc_svc
from app import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run(force: bool = False, nip_filter: str | None = None):
    arc_svc.init_model()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if nip_filter:
                cur.execute(
                    "SELECT nip, foto_wajah, embedding_arcface FROM tbl_siabdi_face_embeddings WHERE nip = %s",
                    (nip_filter,),
                )
            else:
                cur.execute(
                    "SELECT nip, foto_wajah, embedding_arcface FROM tbl_siabdi_face_embeddings WHERE foto_wajah IS NOT NULL"
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    logger.info(f"Total baris ditemukan: {len(rows)}")
    logger.info(f"Membaca foto dari: {config.FOTO_FACE_PATH}")
    ok = skip = fail = 0

    for row in rows:
        nip        = row["nip"]
        foto_name  = row["foto_wajah"]
        has_arcface = bool(row.get("embedding_arcface"))

        if has_arcface and not force:
            logger.info(f"[SKIP] {nip} — sudah ada embedding_arcface.")
            skip += 1
            continue

        # Foto wajah disimpan sama seperti FaceController::register() di Laravel
        # base_path('uploads/foto_face') + filename saja (bukan full path)
        foto_path = os.path.join(config.FOTO_FACE_PATH, foto_name)
        if not os.path.isfile(foto_path):
            logger.warning(f"[FAIL] {nip} — file tidak ditemukan: {foto_path}")
            fail += 1
            continue

        embedding = arc_svc.extract_embedding_from_file(foto_path)
        if embedding is None:
            logger.warning(f"[FAIL] {nip} — wajah tidak terdeteksi di {foto_name}")
            fail += 1
            continue

        saved = arc_svc.save_embedding(nip, embedding)
        if saved:
            logger.info(f"[OK]   {nip} — embedding disimpan.")
            ok += 1
        else:
            logger.error(f"[FAIL] {nip} — gagal UPDATE DB.")
            fail += 1

    logger.info(f"\n=== Selesai === OK:{ok}  SKIP:{skip}  FAIL:{fail}  TOTAL:{len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk extract ArcFace embeddings dari foto_wajah")
    parser.add_argument("--force", action="store_true", help="Overwrite embedding_arcface yang sudah ada")
    parser.add_argument("--nip", type=str, default=None, help="Proses satu NIP saja (untuk testing)")
    args = parser.parse_args()
    run(force=args.force, nip_filter=args.nip)
