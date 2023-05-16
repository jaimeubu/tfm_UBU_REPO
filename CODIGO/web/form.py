import os
from flask import Flask, redirect, request, render_template, url_for
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import math

app = Flask(__name__)
script_route = os.path.dirname(os.path.abspath(__file__))
models = []
models.append(joblib.load(os.path.join(script_route, "logisticReggression.pkl")))
models.append(joblib.load(os.path.join(script_route, "neuralNetwork.pkl")))
models.append(joblib.load(os.path.join(script_route, "randomForest.pkl")))
models.append(joblib.load(os.path.join(script_route, "svm.pkl")))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    highBP = request.form.get("highBP")
    highBP = 1.0 if highBP else 0.0
    highChol = request.form.get("highChol")
    highChol = 1.0 if highChol else 0.0
    cholCheck = request.form.get("cholCheck")
    cholCheck = 1.0 if cholCheck else 0.0
    bmi = request.form.get("bmi")
    smoker = request.form.get("smoker")
    smoker = 1.0 if smoker else 0.0
    stroke = request.form.get("stroke")
    stroke = 1.0 if stroke else 0.0
    diabetes = request.form.get("diabetes")
    physActivity = request.form.get("physActivity")
    physActivity = 1.0 if physActivity else 0.0
    fruits = request.form.get("fruits")
    fruits = 1.0 if fruits else 0.0
    veggies = request.form.get("veggies")
    veggies = 1.0 if veggies else 0.0
    hvyAlcoholConsump = request.form.get("hvyAlcoholConsump")
    hvyAlcoholConsump = 1.0 if hvyAlcoholConsump else 0.0
    anyHealthcare = request.form.get("anyHealthcare")
    anyHealthcare = 1.0 if anyHealthcare else 0.0
    noDocbcCost = request.form.get("noDocbcCost")
    noDocbcCost = 1.0 if noDocbcCost else 0.0
    genHlth = request.form.get("genHlth")
    mentHlth = request.form.get("mentHlth")
    physHlth = request.form.get("physHlth")
    diffWalk = request.form.get("diffWalk")
    diffWalk = 1.0 if diffWalk else 0.0
    sex = request.form.get("sex")
    age = int(request.form.get("age"))
    education = request.form.get("education")
    income = request.form.get("income")

    input_data = {
    'HighBP': [float(highBP)],
    'HighChol': [float(highChol)],
    'CholCheck': [float(cholCheck)],
    'BMI': [float(bmi)],
    'Smoker': [float(smoker)],
    'Stroke': [float(stroke)],
    'Diabetes': [float(diabetes)],
    'PhysActivity': [float(physActivity)],
    'Fruits': [float(fruits)],
    'Veggies': [float(veggies)],
    'HvyAlcoholConsump': [float(hvyAlcoholConsump)],
    'AnyHealthcare': [float(anyHealthcare)],
    'NoDocbcCost': [float(noDocbcCost)],
    'GenHlth': [float(genHlth)],
    'MentHlth': [float(mentHlth)],
    'PhysHlth': [float(physHlth)],
    'DiffWalk': [float(diffWalk)],
    'Sex': [float(sex)],
    'Age': [(float(14) if age > 80 else float(math.floor(((age)/5)-2)))],
    'Education': [float(education)],
    'Income': [float(income)]
    }

    df = pd.DataFrame(input_data)
    #print(df)
    predictions = []
    for model in models:
        if isinstance(model, RandomForestClassifier):
            # Es el único algoritmo que no se beneficia del scaler
            prediction = model.predict(df)[0]
        elif isinstance(model, MLPClassifier):
            loaded_scaler = joblib.load(os.path.join(script_route, 'nNScaler.joblib'))
            prediction = model.predict(loaded_scaler.transform(df))[0]
        elif isinstance(model, LogisticRegression):
            loaded_scaler= joblib.load(os.path.join(script_route, 'lRScaler.joblib'))
            prediction = model.predict(loaded_scaler.transform(df))[0]
        else:
           loaded_scaler = joblib.load(os.path.join(script_route, 'svmScaler.joblib'))
           prediction = model.predict(loaded_scaler.transform(df))[0]
        #prediction = model.predict(df.values)
        predictions.append(prediction)
    print(predictions)
    sum_pred = int(sum(predictions))
    
    if sum_pred == 0:
        result = "Riesgo bajo"
    elif sum_pred <= 2:
        result = "Riesgo alto"
    else:
        result = "Riesgo extremadamente alto"
    
    return redirect(url_for('show_result', result=sum_pred))

@app.route('/result', methods=['GET'])
def show_result():
    result = request.args.get('result', '')
    return render_template('results.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
