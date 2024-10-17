from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Memuat model XGBoost yang telah disimpan
model = xgb.XGBClassifier()
model.load_model('model/diabetes_model.json')

model_columns = pd.read_csv('model/diabetes_columns.csv').squeeze().tolist()


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

        # Membuat DataFrame dengan input user
        input_data = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }])

        # Memastikan kolom input sesuai dengan yang digunakan saat pelatihan
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Melakukan prediksi
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        # Menentukan hasil dan rekomendasi
        if prediction[0] == 1:
            result = f"Anda <span class='red-text'>berisiko tinggi</span> terkena diabetes dengan probabilitas {probability*100:.2f}%."
            recommendation = "Disarankan untuk berkonsultasi dengan dokter dan melakukan perubahan gaya hidup seperti meningkatkan aktivitas fisik dan mengatur pola makan."
        else:
            result = f"Anda <span class='green-text'>berisiko rendah</span> terkena diabetes dengan probabilitas {probability*100:.2f}%."
            recommendation = "Terus jaga gaya hidup sehat dengan pola makan seimbang dan rutin berolahraga."

        return render_template('index.html', prediction=result, probability=probability * 100, recommendation=recommendation)

    except Exception as e:
        return jsonify({'error': str(e)})

# Vercel serverless function requires this
if __name__ == '__main__':
    app.run(debug=True)
