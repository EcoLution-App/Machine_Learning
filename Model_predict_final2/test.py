import requests

# URL tempat aplikasi Flask berjalan
url = 'http://localhost:5000/'

# Data yang akan digunakan untuk uji coba
nama_kabupaten_kota = 'KABUPATEN BANDUNG'

# Request ke aplikasi Flask untuk prediksi
response = requests.post(url, data={'nama_kabupaten_kota': nama_kabupaten_kota})

# Memeriksa apakah respons berhasil atau tidak
if response.ok:
    prediction_data = response.json()
    combined_values_x = prediction_data.get('combined_values_x')
    combined_values_y = prediction_data.get('combined_values_y')
    # Print combined values
    print("Combined Values X:", combined_values_x)
    print("Combined Values Y:", combined_values_y)
else:
    print("Gagal melakukan prediksi. Terjadi kesalahan.")