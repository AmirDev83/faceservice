#!/bin/bash
# Deploy/update faceservice: pull kode terbaru, rebuild image, restart container.
# Jalankan dari server tempat faceservice sudah di-clone: ./deploy.sh
set -e

cd "$(dirname "$0")"

echo "=== Cek perubahan lokal yang belum di-commit ==="
if [ -n "$(git status --porcelain)" ]; then
    echo "ADA perubahan lokal yang belum di-commit di folder ini:"
    git status --short
    echo "Batal — commit/stash dulu perubahan itu, atau hapus manual kalau memang tidak dibutuhkan, sebelum deploy ulang."
    exit 1
fi

echo "=== Pull kode terbaru dari GitHub ==="
git pull

echo "=== Build image (kalau ada perubahan kode/dependency) ==="
docker compose build

echo "=== Restart container dengan image baru ==="
docker compose up -d

echo "=== Bersihkan image lama yang menumpuk (dangling) ==="
docker image prune -f

echo ""
echo "=== Selesai. Log 30 baris terakhir: ==="
sleep 2
docker compose logs --tail=30 faceservice

echo ""
echo "Pantau log realtime: docker compose logs -f faceservice"
