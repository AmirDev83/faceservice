import pymysql
import pymysql.cursors
from dbutils.pooled_db import PooledDB
from app import config

# Pool persisten — sebelumnya tiap panggilan get_connection() buka koneksi TCP+auth
# baru ke MySQL lalu ditutup lagi (conn.close()). Di jam sibuk check-in itu overhead
# signifikan per-request. PooledDB menjaga koneksi tetap terbuka dan dipakai ulang;
# conn.close() dari caller cuma mengembalikan koneksi ke pool (bukan menutup beneran),
# jadi tidak perlu ubah kode pemanggil (arcface.py) sama sekali.
_pool = PooledDB(
    creator=pymysql,
    maxconnections=10,
    mincached=2,
    maxcached=5,
    blocking=True,
    ping=1,  # cek koneksi hidup sebelum dipakai, reconnect otomatis kalau timeout
    host=config.DB_HOST,
    port=config.DB_PORT,
    user=config.DB_USER,
    password=config.DB_PASSWORD,
    database=config.DB_NAME,
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)


def get_connection():
    return _pool.connection()
