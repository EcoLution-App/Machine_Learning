import io
import base64
import requests
import matplotlib.pyplot as plt
from PIL import Image

# URL tempat aplikasi Flask berjalan
url = 'http://localhost:5000/'

# Data yang akan digunakan untuk uji coba
nama_kabupaten_kota = 'KABUPATEN CIANJUR'

# Request ke aplikasi Flask untuk prediksi
response = requests.post(url, data={'nama_kabupaten_kota': nama_kabupaten_kota})

# Memeriksa apakah respons berhasil atau tidak
if response.ok:
    prediction_data = response.json()
    plot_data = prediction_data['plot']

    # Decode string gambar
    image_bytes = base64.b64decode(plot_data)

    # Baca gambar menggunakan PIL (Python Imaging Library)
    img = Image.open(io.BytesIO(image_bytes))

    # Tampilkan gambar menggunakan Matplotlib
    plt.imshow(img)
    plt.axis('off')
    plt.show()
else:
    print("Gagal melakukan prediksi. Terjadi kesalahan.")