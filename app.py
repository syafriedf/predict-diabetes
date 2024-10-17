from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Memuat model yang telah disimpan
with open('model/diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari form
        data = request.form

        # Mengonversi input ke format yang sesuai
        pregnancies = int(data['Pregnancies'])
        glucose = float(data['Glucose'])
        blood_pressure = float(data['BloodPressure'])
        skin_thickness = float(data['SkinThickness'])
        insulin = float(data['Insulin'])
        bmi = float(data['BMI'])
        dpf = float(data['DiabetesPedigreeFunction'])
        age = int(data['Age'])

        # Membuat array fitur
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, dpf, age]])

        # Melakukan prediksi
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        # Menentukan hasil dan rekomendasi
        if prediction[0] == 1:
            result = f"Anda <span class='red-text'>berisiko tinggi</span> terkena diabetes dengan probabilitas {probability*100:.2f}%."
            prob = probability * 100
            recommendation = "Disarankan untuk berkonsultasi dengan dokter dan melakukan perubahan gaya hidup seperti meningkatkan aktivitas fisik dan mengatur pola makan."
        else:
            result = f"Anda <span class='green-text'>berisiko rendah</span> terkena diabetes dengan probabilitas {probability*100:.2f}%."
            prob = probability * 100
            recommendation = "Terus jaga gaya hidup sehat dengan pola makan seimbang dan rutin berolahraga."


        return render_template('index.html', prediction=result, probability=prob, recommendation=recommendation)
    except Exception as e:
        return jsonify({'error': str(e)})

# Vercel serverless function requires this
if __name__ == '__main__':
    app.run(debug=True)
