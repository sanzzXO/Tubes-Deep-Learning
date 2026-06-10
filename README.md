# Tugas Besar Deep Learning

## Deskripsi Proyek

Proyek ini dikembangkan sebagai bagian dari tugas Ujian Tengah Semester (UTS) mata kuliah Deep Learning. Tujuan utama dari proyek ini adalah membangun model deep learning yang mampu memprediksi kondisi lima komponen kendaraan berdasarkan citra yang diperoleh dari model mobil 3D interaktif.

Komponen yang diprediksi meliputi:

* Front Left Door (FL)
* Front Right Door (FR)
* Rear Left Door (RL)
* Rear Right Door (RR)
* Hood

Setiap komponen memiliki dua kemungkinan kondisi:

* 0 = Closed (Tertutup)
* 1 = Open (Terbuka)

Permasalahan ini termasuk ke dalam **multi-label image classification** karena satu gambar dapat memiliki lebih dari satu komponen yang berada dalam kondisi terbuka secara bersamaan.

---

## Struktur Repository

```text
.
├── dataset/
│   ├── images
│   └── labels.csv
│
├── selenium.ipynb
├── modeling_rn.ipynb
├── best_resnet18.pth
└── README.md
```

### Penjelasan File

| File                | Deskripsi                                                                      |
| ------------------- | ------------------------------------------------------------------------------ |
| `selenium.ipynb`    | Notebook untuk proses pengumpulan dataset secara otomatis menggunakan Selenium |
| `modeling_rn.ipynb` | Notebook untuk preprocessing, pelatihan model, evaluasi, dan prediksi          |
| `best_resnet18.pth` | Model ResNet18 terbaik hasil pelatihan                                         |
| `dataset/`          | Dataset gambar-gambar kendaraan beserta labelnya                               |

