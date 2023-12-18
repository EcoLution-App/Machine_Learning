import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model_predict.h5')
# Load the dataset for plotting
df_kota_banjir = pd.read_csv('df_kota_banjir.csv')
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler with the numerical data (assuming 'tahun' is the numerical column)
scaler.fit(df_kota_banjir[['tahun']])
# Initialize the LabelEncoder for city names
label_encoder = LabelEncoder()
# Fit the LabelEncoder with the categorical data
label_encoder.fit(df_kota_banjir['nama_kabupaten_kota'])

# Define an endpoint for predictions and plotting
@app.route('/', methods=['GET', 'POST'])
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
        prediction_years = [last_year + i for i in range(1, 4)]  # Predict for the next 3 years
        scaled_years = scaler.transform([[year] for year in prediction_years])[:, 0]

        # Make predictions for the next 3 years
        try:
            prediction_inputs = np.column_stack((np.repeat(encoded_kabupaten_kota, 3), scaled_years))
            predicted_values = model.predict(prediction_inputs).flatten().tolist()

            # Convert prediction_years to str to avoid numpy.int64 keys error
            prediction_years_str = list(map(str, prediction_years))
            prediction_result = dict(zip(prediction_years_str, predicted_values))

            # Plotting the actual and predicted values
            plt.figure(figsize=(8, 6))
            plt.plot(actual_values['tahun'], actual_values['jumlah_banjir'], label='Aktual')
            plt.plot(prediction_years, predicted_values, 'ro--', label='Prediksi')
            plt.xlabel('Tahun')
            plt.ylabel('Jumlah Kejadian Banjir')
            plt.title(f'Prediksi Banjir di {nama_kabupaten_kota} 2 Tahun Mendatang')
            plt.legend()
            plt.grid()

            # Save the plot to a BytesIO object
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_data = base64.b64encode(img.getvalue()).decode()
            img.close()

            return jsonify({'predictions': prediction_result, 'plot': plot_data})
        except Exception as e:
            return jsonify({'error': f'Gagal melakukan prediksi: {str(e)}'})

@app.route('/plot')
def plot():
    img_data = request.args.get('img')
    return send_file(io.BytesIO(base64.b64decode(img_data)), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)