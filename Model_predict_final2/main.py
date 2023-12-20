import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model, dataset, and preprocessors
model = load_model('model_predict.h5')
df_kota_banjir = pd.read_csv('https://storage.googleapis.com/ecolution/dataset/df_kota_banjir.csv')
scaler = StandardScaler()
scaler.fit(df_kota_banjir[['tahun']])
label_encoder = LabelEncoder()
label_encoder.fit(df_kota_banjir['nama_kabupaten_kota'])

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get the city name from the form
        nama_kabupaten_kota = request.form.get('nama_kabupaten_kota')

        # Ensure the city name is provided
        if not nama_kabupaten_kota:
            return jsonify({'error': 'Nama kota tidak diberikan'})

        # Preprocess the city name with LabelEncoder
        try:
            encoded_kabupaten_kota = label_encoder.transform([nama_kabupaten_kota])
        except ValueError as e:
            return jsonify({'error': f'Nama kota tidak valid: {str(e)}'})

        # Get historical data for the selected city
        actual_values = df_kota_banjir[df_kota_banjir['nama_kabupaten_kota'] == nama_kabupaten_kota]

        # Prepare input for prediction (for 3 years in the future)
        if actual_values.empty:
            return jsonify({'error': 'Nama kota tidak ditemukan'})

        last_year = actual_values['tahun'].max()
        prediction_years = [last_year + i for i in range(1, 4)]  # Predict for 2023-2025
        scaled_years = scaler.transform([[year] for year in prediction_years])[:, 0]

        # Make predictions for 2023-2025
        try:
            prediction_inputs = np.column_stack((np.repeat(encoded_kabupaten_kota, 3), scaled_years))
            predicted_values = model.predict(prediction_inputs).flatten().tolist()

            # Save the values to variables
            actual_values_x = actual_values['tahun'].tolist()
            actual_values_y = actual_values['jumlah_banjir'].tolist()
            predicted_values_x = list(map(int, prediction_years))
            predicted_values_y = predicted_values

            # Combine actual and predicted values
            combined_values_x = actual_values_x + predicted_values_x
            combined_values_y = actual_values_y + predicted_values_y

            return jsonify({'combined_values_x': combined_values_x,
                            'combined_values_y': combined_values_y})

        except Exception as e:
            return jsonify({'error': f'Gagal melakukan prediksi: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=False)