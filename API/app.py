import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
import pickle
import math
import base64
from flask import jsonify

app= Flask(__name__)

@app.route('/predict/<ID>', methods = ['GET'])
def predict(ID):
    '''
    For rendering results on HTML GUI
    '''

    url_df = 'https://raw.githubusercontent.com/jlu0915/P7/master/API/df_API.csv'
    df_test = pd.read_csv(url_df)
    #df_test = pd.read_csv('df_API.csv')
    liste_clients = list(df_test['SK_ID_CURR'].unique())
    
    probability_default_payment = 0
    model = pickle.load(open('LGBM.pickle', 'rb')).best_estimator_
    seuil = 0.92
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

    return jsonify({"prediction": prediction, "score": probability_default_payment.tolist()})

#app.run(port=8080, debug=True)
#app.run(debug=True)
app.run(port=8080)
'''
if __name__ == "__main__":
    """
    Run app.
    Note : comment this line if you want to deploy on heroku
    """
    app.run(port=8080, debug=True)
'''
