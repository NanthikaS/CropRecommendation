'''
create a folder
open them in vs code
create a virtual environment using 'python -m venv venv'
if running scripts is disabled then use
1) click windows
2)type powershell
3)execute the below command
 Get-ExecutionPolicy
>> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted -Force
activate the environment using '.\venv\Scripts\activate'
Install the flask in the folder using 'python -m pip install flask'
run the app using 'python -m flask --app .\app.py run'
or
run the app using 'python -m flask run'
'''
# from flask import Flask
# app=Flask(__name__)

# #for url routing
# @app.route('/a')

# def home():
#     return "Hello worms"

import numpy as np
from flask import Flask, url_for,request, jsonify, render_template
import math
import joblib
import pickle
app = Flask(__name__)
modl = pickle.load(open('rfmodel.pkl','rb'))
# modl = joblib.load('bag_model.joblib')
@app.route('/')
def Home():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    n = request.form['N']
    p = request.form['P']
    k = request.form['K']
    tem = float(request.form['temperature'])
    hum = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rain = float(request.form['rainfall'])
    values = np.array([[n,p,k,tem,hum,ph,rain]])
    
    result = modl.predict(values)
    return render_template('rec.html',predicted_crop=result)

if __name__=="__main__":
    app.run()