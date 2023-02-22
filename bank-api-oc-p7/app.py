import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
import pickle
import math
import base64



app= Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST', 'GET'])

def predict():
    '''
    For rendering results on HTML GUI
    '''

    url_df = 'https://raw.githubusercontent.com/jlu0915/P7/master/API/df_API.csv'
    df_test = pd.read_csv(url_df)
    #df_test = pd.read_csv('df_API.csv')
    liste_clients = list(df_test['SK_ID_CURR'].unique())

    model = pickle.load(open('LGBM.pickle', 'rb')).best_estimator_
    seuil = 0.92

    ID = request.form['id_client']
    ID = int(ID)
    if ID not in liste_clients:
        prediction="Ce client n'est pas répertorié"
    else :
        X = df_test[df_test['SK_ID_CURR'] == ID]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        #data = df[df.index == comment]
        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Prêt NON Accordé"
        else:
            prediction = "Prêt Accordé"

    return render_template('index.html', prediction_text=prediction)

# Define endpoint for flask
app.add_url_rule('/predict', 'predict', predict)


# Run app.
# Note : comment this line if you want to deploy on heroku
app.run()
#app.run(debug=True)
