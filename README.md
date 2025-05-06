# UTS - Data Mining 

Proyek ini bertujuan untuk **memprediksi harga penutupan saham (Close)** menggunakan algoritma **Decision Tree Regressor**, salah satu metode dalam pembelajaran mesin (machine learning). Dataset yang digunakan diambil dari Kaggle dengan nama `stocknews`. Proyek ini mencakup tahap mulai dari pengunduhan data, pembersihan data, pelatihan model, evaluasi, hingga visualisasi hasil.

---

Penjelasan Program

Program dimulai dengan mengimpor beberapa library penting:
- `kagglehub` untuk mengunduh dataset dari Kaggle.
- `os` dan `pandas` untuk mengelola file dan membaca data.
- `sklearn` untuk membagi dataset dan membangun model prediksi.
- `matplotlib` untuk menampilkan grafik pohon keputusan.

Kemudian, dataset bernama `stocknews` diunduh menggunakan `kagglehub`, dan data CSV yang relevan (`upload_DJIA_table.csv`) dibaca ke dalam DataFrame. Setelah ditampilkan isi awal dan struktur kolomnya, data dibersihkan dengan cara menghapus baris yang memiliki nilai kosong pada kolom penting seperti `Open`, `High`, `Low`, `Close`, `Volume`, dan `Adj Close`.

Fitur yang digunakan untuk prediksi adalah:
- `Open`
- `High`
- `Low`
- `Volume`
- `Adj Close`

Sementara target yang ingin diprediksi adalah:
- `Close` (harga penutupan saham)

Data dibagi menjadi dua bagian:
- 90% sebagai **data latih**
- 10% sebagai **data uji**

Model yang digunakan adalah **Decision Tree Regressor** dengan kedalaman maksimum 3. Setelah model dilatih, performanya diukur menggunakan **R² score**. Akhirnya, pohon keputusan divisualisasikan untuk melihat bagaimana model mengambil keputusan berdasarkan data input.

---

 Kode Program

```python
import kagglehub  # Untuk mengunduh dataset dari Kaggle
import os  # Untuk mengelola path dan file
import pandas as pd  # Untuk membaca dan memproses data
from sklearn.model_selection import train_test_split  # Untuk membagi data latih dan uji
from sklearn.tree import DecisionTreeRegressor, plot_tree  # Model Decision Tree untuk regresi dan visualisasi
import matplotlib.pyplot as plt  # Untuk menampilkan grafik

# Mengunduh dataset dari Kaggle
path = kagglehub.dataset_download("aaron7sun/stocknews")  
print("Path to dataset files:", path)

# Menentukan folder tempat dataset disimpan
dataset_folder = '/root/.cache/kagglehub/datasets/aaron7sun/stocknews/versions/2'

# Menampilkan nama-nama file di dalam folder dataset
for filename in os.listdir(dataset_folder):
    print(filename)

# Menyusun path lengkap ke file CSV yang ingin dibaca
file_path = os.path.join(dataset_folder, 'upload_DJIA_table.csv')

# Membaca file CSV ke dalam DataFrame
df = pd.read_csv(file_path)

# Menampilkan 5 baris pertama dari dataset
print(df.head())

# Menampilkan nama-nama kolom pada dataset
print(df.columns)

# Menghapus baris yang memiliki nilai kosong pada kolom penting
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])

# Menentukan fitur (X) yang akan digunakan untuk prediksi
x = df[['Open', 'High', 'Low', 'Volume', 'Adj Close']]  # Fitur input

# Menentukan target (y) yang akan diprediksi, yaitu harga penutupan
y = df['Close']  # Target output

# Membagi data menjadi data latih dan data uji (90% latih, 10% uji)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Membuat model pohon keputusan untuk regresi dengan kedalaman maksimum 3
dtree = DecisionTreeRegressor(max_depth=3)
dtree.fit(x_train, y_train)  # Melatih model

# Mengukur performa model terhadap data uji menggunakan R² score
accuracy = dtree.score(x_test, y_test)
print(f"Akurasi model: {accuracy*100:.2f}%")

# Menampilkan grafik pohon keputusan
plt.figure(figsize=(30, 20))
plot_tree(dtree, filled=True)
plt.show()
