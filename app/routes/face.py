"""
Endpoints:
  POST /verify  — terima foto + NIP dari Laravel, kembalikan {verified, score}
  POST /extract — ekstrak + simpan embedding ArcFace dari foto (Phase 2 / admin)
  GET  /health  — cek apakah service berjalan dan model sudah dimuat
"""

import time
import asyncio
import logging
import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile, HTTPException

from app import config
from app.services import arcface as arc_svc

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    # get_arcface_threshold() query DB (dgn cache 60s) — sama persis fungsi yang
    # dipakai /verify. Sebelumnya di sini cuma tampilkan config.ARCFACE_THRESHOLD
    # (fallback .env statis), jadi tidak pernah mencerminkan nilai asli dari
    # tbl_siabdi_face_config yang benar-benar dipakai saat verifikasi.
    threshold = await asyncio.to_thread(arc_svc.get_arcface_threshold)
    return {
        "status": "ok",
        "model": "buffalo_l",
        "threshold": threshold,
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

    Foto dikirim ke endpoint internal apisimak untuk disimpan sebagai audit trail
    (agar bisa dievaluasi dari SIMAK) — lihat _upload_foto_ke_apisimak().
    """
    image_bytes = await foto.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Foto kosong.")

    # Upload di background — tidak perlu ditunggu sebelum verdict verify dikirim
    # ke Laravel/Flutter, ini murni audit trail. Kalau gagal (mis. apisimak belum
    # punya endpoint internal-nya / jaringan bermasalah), verifikasi TETAP jalan
    # normal — cuma foto audit untuk kasus itu yang tidak tersimpan.
    background_tasks.add_task(
        _upload_foto_ke_apisimak, nip, image_bytes, "verify", "audit_selfie"
    )

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
      - FaceController::register() di Laravel — dipanggil OTOMATIS setelah foto
        master SUDAH tersimpan di apisimak (lihat FaceController.php baris
        ~110-124). Endpoint ini TIDAK menyimpan foto sama sekali — cuma
        ekstraksi embedding — karena pemanggilnya sudah punya salinan foto
        master sendiri. Kalau di sini foto ikut disimpan (upload balik ke
        apisimak), hasilnya 2 file duplikat untuk 1 registrasi yang sama.
      - scripts/extract_embeddings.py (Phase 2, baca file lokal langsung,
        tidak lewat endpoint HTTP ini sama sekali)
      - Admin/operator lewat HTTP untuk 1 NIP (asumsi foto master sudah ada
        di apisimak dari alur registrasi normal)
    """
    image_bytes = await foto.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Foto kosong.")

    embedding = await asyncio.to_thread(arc_svc.extract_embedding, image_bytes)
    if embedding is None:
        return {"success": False, "message": f"[{nip}] Wajah tidak terdeteksi di foto."}

    saved = await asyncio.to_thread(arc_svc.save_embedding, nip, embedding)
    if not saved:
        return {"success": False, "message": f"[{nip}] NIP tidak ditemukan di database."}

    return {
        "success": True,
        "message": f"[{nip}] Embedding ArcFace berhasil disimpan.",
    }


async def _upload_foto_ke_apisimak(
    nip: str, image_bytes: bytes, prefix: str, kind: str
) -> bool:
    """
    Kirim foto ke endpoint internal apisimak (POST APISIMAK_INTERNAL_URL) untuk
    disimpan ke folder uploads/{kind}/ di server-app — menggantikan tulis file
    lokal langsung, karena faceservice & apisimak sekarang bisa di host berbeda
    (server-utama vs server-app) tanpa filesystem yang dibagi.

    Non-fatal: kegagalan di sini (network/endpoint belum ada) TIDAK BOLEH
    menggagalkan hasil verifikasi/registrasi wajah — foto cuma untuk audit trail.
    """
    if not config.APISIMAK_INTERNAL_URL:
        logger.warning(f"[{nip}] APISIMAK_INTERNAL_URL belum diset, lewati upload foto.")
        return False

    filename = f"{nip}_{prefix}_{int(time.time())}.jpg"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                config.APISIMAK_INTERNAL_URL,
                headers={"Authorization": f"Bearer {config.APISIMAK_INTERNAL_TOKEN}"},
                data={"nip": nip, "kind": kind, "filename": filename},
                files={"foto": (filename, image_bytes, "image/jpeg")},
            )
        if resp.status_code != 200:
            logger.warning(f"[{nip}] Upload foto ke apisimak gagal: HTTP {resp.status_code} {resp.text[:200]}")
            return False
        logger.info(f"[{nip}] Foto {kind} terkirim ke apisimak: {filename}")
        return True
    except httpx.HTTPError as e:
        logger.warning(f"[{nip}] Upload foto ke apisimak gagal (network): {e}")
        return False
