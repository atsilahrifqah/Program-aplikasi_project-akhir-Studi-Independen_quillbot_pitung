from flask import Flask, jsonify, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
pickle_ln = open('model.pkl', 'rb')
model = pickle.load(pickle_ln)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("form.html")


@app.route("/predict", methods=['POST'])
def predict():

    age = float(request.form['age'])
    sex = float(request.form['sex'])
    chest_pain_type = float(request.form['chest_pain_type'])
    resting_bp = float(request.form['resting_bp'])
    cholesterol = float(request.form['cholesterol'])
    fasting_bs = float(request.form['fasting_bs'])
    resting_ECG = float(request.form['resting_ECG'])
    maxHR = float(request.form['maxHR'])
    exercise_angina = float(request.form['exercise_angina'])
    old_peak = float(request.form['old_peak'])
    ST_slope = float(request.form['ST_slope'])

# Print or log the input data for debugging
    print("Input Data:", [age, sex, chest_pain_type, resting_bp, cholesterol,
          fasting_bs, resting_ECG, maxHR, exercise_angina, old_peak, ST_slope])

    arr = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol,
                    fasting_bs, resting_ECG, maxHR, exercise_angina, old_peak, ST_slope]])

    # Print or log the input array for debugging
    print("Input Array:", arr)

    prediction = model.predict(arr)

    if prediction == 0:
        hasil = "Anda Terkena Penyakit Jantung"
        return render_template("form.html", hasil=hasil)
    else:
        hasil = "Anda Tidak Terkena Penyakit Jantung"
        return render_template("form.html", hasil=hasil)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
