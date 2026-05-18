# Expense Forecasting API — Deploy ke Railway

## Struktur Folder
```
railway_deploy/
├── main.py               ← kode FastAPI
├── requirements.txt      ← library yang dibutuhkan
├── Procfile              ← cara Railway jalankan server
├── saved_model/          ← folder ini kamu buat sendiri
│   ├── expense_forecasting_model.keras
│   ├── scaler.pkl
│   └── user_to_idx.pkl
└── README.md
```

---

## Langkah Deploy ke Railway

### 1. Siapkan folder di komputer lokal
Buat folder baru, lalu taruh semua file ini di dalamnya:
- `main.py`
- `requirements.txt`
- `Procfile`
- Folder `saved_model/` berisi 3 file model (download dari Google Drive / Colab)

### 2. Push ke GitHub
```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```

### 3. Deploy di Railway
1. Buka https://railway.app → Login dengan GitHub
2. Klik **New Project** → **Deploy from GitHub repo**
3. Pilih repo yang baru kamu push
4. Railway otomatis detect Python dan mulai deploy
5. Tunggu proses build selesai (~3-5 menit)

### 4. Dapatkan URL permanen
1. Di Railway dashboard → klik project kamu
2. Klik tab **Settings** → **Networking**
3. Klik **Generate Domain**
4. Kamu dapat URL permanen seperti: `https://expense-api.up.railway.app`

---

## Test API setelah deploy
Buka browser:
```
https://expense-api.up.railway.app/docs
```

---

## Endpoint
| Method | URL | Fungsi |
|--------|-----|--------|
| GET | `/` | Health check |
| POST | `/predict/csv` | Upload CSV → prediksi |
| POST | `/predict/json` | Kirim JSON → prediksi |

---

## Catatan Penting
- Railway gratis tier: **500 jam/bulan** — cukup untuk development
- File `saved_model/` WAJIB ada di dalam folder sebelum push ke GitHub
- Pakai `tensorflow-cpu` bukan `tensorflow` biasa agar lebih ringan di server
