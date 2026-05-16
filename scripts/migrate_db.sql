-- Phase 2: Tambah kolom embedding_arcface ke tbl_siabdi_face_embeddings
-- Database: db_simpeg (sama dengan Laravel DB_DATABASE)
--
-- Jalankan di server:
--   mysql -h 127.0.0.1 -u adminsimak -p db_simpeg < scripts/migrate_db.sql

USE db_simpeg;

ALTER TABLE tbl_siabdi_face_embeddings
    ADD COLUMN IF NOT EXISTS embedding_arcface LONGTEXT NULL
        COMMENT 'ArcFace buffalo_l 512-dim embedding (JSON float32 array), diekstrak dari foto_wajah'
        AFTER embedding_frontal;

-- Verifikasi
SELECT
    COUNT(*)                                         AS total_pegawai,
    SUM(foto_wajah IS NOT NULL)                      AS punya_foto,
    SUM(embedding_arcface IS NOT NULL)               AS sudah_arcface,
    SUM(foto_wajah IS NOT NULL AND embedding_arcface IS NULL) AS perlu_extract
FROM tbl_siabdi_face_embeddings;
