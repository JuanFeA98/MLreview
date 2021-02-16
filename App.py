from flask import Flask, render_template, flash, request, redirect, url_for
from algoritmos import regLineal, regLogistica, treeDecision, kmeans
import numpy as np

app = Flask(__name__)

app.secret_key = 'mysecretkey'

@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/LinealRegression')
def RegLineal():
    return render_template('regLin.html')

@app.route('/calcRegLin', methods=['POST'])
def calcRegLin():
    if request.method == 'POST':
        firstFloor = float(request.form['firstFloor'])
        secondFloor = float(request.form['secondFloor'])
        overQuall = float(request.form['overQuall'])
        garage = float(request.form['garage'])
        coeficientes = regLineal()
        resultado = (round(coeficientes[0] + (firstFloor * coeficientes[1]) + (secondFloor * coeficientes[2]) + (overQuall * coeficientes[3]) + (garage * coeficientes[4]), 2))

        flash(f'El resultado estimado es de: {resultado} dólares.')
        return redirect(url_for('RegLineal'))

@app.route('/LogisticRegression')
def RegLogistica():
    return render_template('regLog.html')

@app.route('/calcRegLog', methods=['POST'])
def calcRegLog():
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodPressure = float(request.form['bloodPressure'])
        insulin = float(request.form['insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['age'])

        data = np.array([[pregnancies, glucose, bloodPressure, insulin, BMI, DiabetesPedigreeFunction, age]])

        resultado = regLogistica(data)

        if resultado == 1:
            flash('El paciente tiene diabetes.')
        else:
            flash('El paciente no tiene diabetes.')

        return redirect(url_for('RegLogistica'))

@app.route('/TreeDecision')
def TreeDecision():
    coeficientes = regLineal()
    resultado = (round(coeficientes[0] + (10 * coeficientes[1]), 2))
    return render_template('treeDes.html', resultado = resultado)

@app.route('/calcTree', methods=['POST'])
def calcTree():
    if request.method == 'POST':
        sex = float(request.form['sex'])
        pclass = float(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = float(request.form['sibsp'])
        parch = float(request.form['parch'])
        fare = float(request.form['fare'])

        data = np.array([[sex, pclass, age, sibsp, parch, fare]])

        resultado = treeDecision(data)

        if resultado == 1:
            flash('El pasajero sobrevivio.')
        else:
            flash('El pasejor no sobrevivo.')

        return redirect(url_for('TreeDecision'))


@app.route('/Kmeans')
def Kmeans():
    return render_template('kmeans.html')

@app.route('/calcKmeans', methods=['POST'])
def calcKmeans():
    if request.method == 'POST':
        sepalL = float(request.form['sepalL'])
        sepalW = float(request.form['sepalW'])
        petalL = float(request.form['petalL'])
        petalW = float(request.form['petalW'])

        data = np.array([[sepalL, sepalW, petalL, petalW]])

        resultado = kmeans(data)

        if resultado == 0:
            flash('La flor es de la clase Setosa.')
        elif resultado == 1:
            flash('La flor es de la clase Versicolor.')
        else:
            flash('La flor es de la clase Virginica.')

        return redirect(url_for('Kmeans'))

if __name__ == "__main__":
    app.run(port=4000, debug=True)