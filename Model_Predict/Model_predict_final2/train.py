import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load dataset
df_kota_banjir = pd.read_csv('df_kota_banjir.csv')

# Prepare features and target
X = df_kota_banjir[['nama_kabupaten_kota', 'tahun']]
y = df_kota_banjir['jumlah_banjir']

# Encode categorical features
label_encoder = LabelEncoder()
X['nama_kabupaten_kota'] = label_encoder.fit_transform(X['nama_kabupaten_kota'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train['tahun'] = scaler.fit_transform(X_train[['tahun']])
X_test['tahun'] = scaler.transform(X_test[['tahun']])

# Create TensorFlow model
model = Sequential([Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
                    Dropout(0.5),
                    Dense(256, activation='relu'),
                    Dropout(0.5),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(1)])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model with early stopping
early_stopping = EarlyStopping(patience=25, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=1000, batch_size=512, 
                    validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])

# Make predictions
predictions = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE) on test set: {mse}")

# Save the trained model to h5 format
nama_file_model_h5 = "model_predict.h5"
model.save(nama_file_model_h5)