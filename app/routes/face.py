"""
Endpoints:
  POST /verify  — terima foto + NIP dari Laravel, kembalikan {verified, score}
  POST /extract — ekstrak + simpan embedding ArcFace dari foto (Phase 2 / admin)
  GET  /health  — cek apakah service berjalan dan model sudah dimuat
"""

import os
import time
import asyncio
import logging
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile, HTTPException

from app import config
from app.services import arcface as arc_svc

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model": "buffalo_l",
        "threshold": config.ARCFACE_THRESHOLD,
        "foto_face_path": config.FOTO_FACE_PATH,
    }


@router.post("/verify")
async def verify(
    background_tasks: BackgroundTasks,
    nip: str = Form(...),
    foto: UploadFile = File(...),
):
    """
    Dipanggil dari Laravel FaceController::verifyLive() saat FACE_SERVICE_ACTIVE=true.

    Alur:
      1. Terima foto selfie (bytes JPEG) + NIP dari Laravel
      2. InsightFace buffalo_l: ekstrak ArcFace embedding dari foto
      3. Banding probe vs embedding_arcface di tbl_siabdi_face_embeddings (db_simpeg)
      4. Kembalikan {verified, score} → Laravel teruskan ke Flutter (use_device=false)

    Foto disimpan sementara di AUDIT_SELFIE_PATH/{nip}_verify_{timestamp}.jpg
    agar bisa dievaluasi dari SIMAK (sama pola dengan storePendingPresence Laravel).
    """
    image_bytes = await foto.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Foto kosong.")

    # Tulis file di background — tidak perlu ditunggu sebelum verdict verify dikirim
    # ke Laravel/Flutter, ini murni audit trail.
    background_tasks.add_task(_save_audit_foto, nip, image_bytes, "verify")

    # extract_embedding (InsightFace, CPU-bound) & verify_against_stored (query DB) itu
    # blocking sync. Dijalankan langsung di sini (dalam "async def") akan mengunci
    # SATU-SATUNYA event loop worker ini — request lain (termasuk /health) ikut
    # tertahan sampai selesai. asyncio.to_thread melepas keduanya ke thread terpisah;
    # ONNX Runtime melepas GIL saat inference native-nya jalan, jadi request lain bisa
    # diproses bersamaan alih-alih antre satu-satu.
    probe = await asyncio.to_thread(arc_svc.extract_embedding, image_bytes)
    if probe is None:
        logger.info(f"[{nip}] Wajah tidak terdeteksi di foto selfie.")
        return {"verified": False, "score": 0.0}

    result = await asyncio.to_thread(arc_svc.verify_against_stored, nip, probe)

    if result.get("detail") == "no_embedding":
        # embedding_arcface belum ada → minta Flutter fallback ke device
        # Ini tidak seharusnya terjadi jika Phase 2 sudah dijalankan
        logger.error(f"[{nip}] embedding_arcface belum ada. Jalankan scripts/extract_embeddings.py")
        raise HTTPException(
            status_code=404,
            detail=f"embedding_arcface belum tersedia untuk NIP {nip}.",
        )

    return {"verified": result["verified"], "score": result["score"]}


@router.post("/extract")
async def extract(
    nip: str = Form(...),
    foto: UploadFile = File(...),
):
    """
    Ekstrak embedding ArcFace dari foto dan simpan ke tbl_siabdi_face_embeddings.

    Dipakai oleh:
      - scripts/extract_embeddings.py (Phase 2, bulk dari foto_wajah di disk)
      - Admin/operator jika ingin update embedding satu NIP via HTTP

    Foto disimpan ke uploads/foto_face/ (sama seperti FaceController::register() di Laravel)
    agar foto_wajah di DB tetap bisa diakses dari SIMAK.
    """
    image_bytes = await foto.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Foto kosong.")

    # Simpan foto ke uploads/foto_face/ dengan pola sama seperti Laravel
    filename = f"{nip}_master_{int(time.time())}.jpg"
    foto_path = os.path.join(config.FOTO_FACE_PATH, filename)
    os.makedirs(config.FOTO_FACE_PATH, exist_ok=True)
    with open(foto_path, "wb") as f:
        f.write(image_bytes)
    logger.info(f"[{nip}] Foto master disimpan: {foto_path}")

    embedding = await asyncio.to_thread(arc_svc.extract_embedding, image_bytes)
    if embedding is None:
        # Hapus foto jika wajah tidak terdeteksi
        try: os.remove(foto_path)
        except OSError: pass
        return {"success": False, "message": f"[{nip}] Wajah tidak terdeteksi di foto."}

    saved = await asyncio.to_thread(arc_svc.save_embedding, nip, embedding)
    if not saved:
        return {"success": False, "message": f"[{nip}] NIP tidak ditemukan di database."}

    return {
        "success": True,
        "message": f"[{nip}] Embedding ArcFace berhasil disimpan.",
        "foto": filename,
    }


def _save_audit_foto(nip: str, image_bytes: bytes, prefix: str = "verify") -> None:
    """
    Simpan foto audit ke uploads/audit_selfie/ dengan pola sama seperti
    PresenceController::storePendingPresence() di Laravel.
    Hanya untuk keperluan audit/evaluasi dari SIMAK, tidak dipakai untuk verifikasi.
    """
    try:
        os.makedirs(config.AUDIT_SELFIE_PATH, exist_ok=True)
        filename = f"{nip}_{prefix}_{int(time.time())}.jpg"
        filepath = os.path.join(config.AUDIT_SELFIE_PATH, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        logger.info(f"[{nip}] Audit foto disimpan: {filepath}")
    except OSError as e:
        logger.warning(f"[{nip}] Gagal simpan audit foto: {e}")
