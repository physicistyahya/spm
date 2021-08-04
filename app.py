import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)
loaded_model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


def predict(input_features):
    global loaded_model
    result = loaded_model.predict(input_features)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        input_features = request.form.to_dict()
        features = []
        features.append(input_features["age"])
        features.append(input_features["hypertension"])
        features.append(input_features["heart_disease"])
        features.append(input_features["avg_glucose_level"])
        features.append(input_features["bmi"])
        features.append(input_features["smoking_status"])
        if input_features["gender"] == 1:
            features += [0, 1]
        else:
            features += [1, 0]
        if input_features["ever married"] == 1:
            features += [0, 1]
        else:
            features += [1, 0]
        if input_features["work_type"] == 0:
            features += [0, 0, 1, 0, 0]
        elif input_features["work_type"] == 1:
            features += [0, 0, 0, 1, 0]
        elif input_features["work_type"] == 2:
            features += [1, 0, 0, 0, 0]
        elif input_features["work_type"] == 3:
            features += [0, 0, 0, 0, 1]
        else:
            features += [0, 1, 0, 0, 0]
        features = np.array(features).reshape(1, 15)
        result = predict(features)
        if int(result) == 1:
            prediction = 'You might want to consider visiting a doctor'
        else:
            prediction = 'No worries! You are as healthy as a horse ;)'
        return render_template("result.html", prediction=prediction)
