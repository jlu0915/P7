import requests
import streamlit as st
import pandas as pd
import pickle
import warnings
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from streamlit_shap import st_shap

warnings.filterwarnings('ignore')
st.set_page_config(page_title='Loan Scoring APP', layout="wide")

sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
'''

st.markdown(sysmenu, unsafe_allow_html=True)

df_dashboard_url = "https://raw.githubusercontent.com/jlu0915/P7/master/API/df_API.csv"
df_url = "https://raw.githubusercontent.com/jlu0915/P7/master/Dashboard/df_data_bis.csv"
df = pd.read_csv(df_url)
df_dashboard = pd.read_csv(df_dashboard_url)
df_dashboard.drop("TARGET", inplace=True, axis=1)
model = pickle.load(open('LGBM.pickle', 'rb')).best_estimator_

#df_dashboard_bis = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/Dashboard/df_dashboard_lite.csv"
#df = pd.read_csv(df_dashboard_bis)

# Récupération de ID_client
#ID_client = st.sidebar.selectbox("Sélectionnez l'ID client", pd.unique(df['SK_ID_CURR']))



def feature_engineering(df):
    new_df = pd.DataFrame()
    new_df = df.copy()
    new_df['CODE_GENDER'] = df['CODE_GENDER'].apply(lambda x: 'Femme' if x == 1 else 'Homme')
    new_df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x : -x/365.25)
    new_df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x : -x/365.25)
    new_df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    new_df['NAME_EDUCATION_TYPE_Highereducation'] = df['NAME_EDUCATION_TYPE_Highereducation'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    return new_df

def get_english_var(var_fr):
    liste_var_en = ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS',
                    'INCOME_PER_PERSON', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation', 'AMT_GOODS_PRICE', 'AMT_CREDIT',
                    'PAYMENT_RATE', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    liste_var_fr = ['ID_Client', 'Genre', 'Âge', 'Ancienneté de l\'emploi', 'Revenus totaux', 'Nombre de personnes dans la famille',
                    'Revenus par personne', 'Voiture personnelle', 'Education secondaire',
                    'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit', 'Fréquence de paiement',
                    'Montant des annuités', 'Source externe 2', 'Source externe 3']
    ind = liste_var_fr.index(var_fr)
    var_en = liste_var_en[ind]
    return var_en

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

def predict():
    col1, col2 = st.columns([3.5, 6.5])
    with col2:
        st.title('_solvency analysis_')
    flag.drop(['SK_ID_CURR'], axis=1, inplace=True)

    url = f"https://bank-api-oc-p7.herokuapp.com/predict/{option}"
    result = requests.get(url, verify=False)
    result.encoding = "utf-8"
    probability_default_payment = result.json()
    
    if probability_default_payment['prediction'] == 'Prêt Accordé':
        accord_credit = "Oui" 
    else:
        accord_credit = "Non" 
        
    score = probability_default_payment['score']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score[0] * 100,
        title={'text': "Client score"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(yaxis={'range': [0, 100]})

    # dataframe filter
    #df_client = df[df['SK_ID_CURR'] == ID_client]
    #new_df = feature_engineering(option)

    #shap.initjs()
    #explainer = shap.TreeExplainer(model)
    #shap_values = explainer.shap_values(flag)

    #shap.force_plot(explainer.expected_value[1], shap_values[1], flag,link='logit', show=False)
    #plt.savefig('explaination-local.png', dpi=100)

    #shap.summary_plot(shap_values, flag, show=False)
    #plt.savefig('features importance.png', dpi=100)
    # st.write(flag.shape)

    #shap.initjs()

    # Shap values
    explainer = shap.TreeExplainer(model)
    #df_api_url = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/Dashboard/df_API_lite.csv"
    #df_API = pd.read_csv(df_api_url)
    df_shap = df_dashboard.loc[:, df_dashboard.columns != 'SK_ID_CURR']
    shap_values = explainer.shap_values(df_shap)
    

    tab1, tab2, tab3 = st.tabs(['Score client', 'Explication du score', 'Comparaison aux autres clients'])
    with tab1:
        if score[0] >= 0.92:
            prediction = "Prêt Accordé"
        else:
            prediction = "Prêt NON Accordé"
        st.write('Probability threshold：{}%'.format(round(score[0]*100, 2)))
        st.write('Predict：{}'.format(prediction))
        st.plotly_chart(fig, use_container_width=True)
    
    
    with tab2:
        #st.image('features importance.png')
        #st.image('explaination-local.png')

        # Interprétation pour l'individu choisi
        st.header("Impact des variables sur le score pour le client " + str(option))
        st.write('Les variables en rose ont contribué à accorder le crédit (donc à augmenter le score).\n Les variables en bleu ont contribué à refuser le crédit (donc à diminuer le score)')
        id = df_dashboard[df_dashboard['SK_ID_CURR'] == option].index
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][id, :], df_shap.iloc[id, :], link='logit'))
        

        st.header("Impact des variables pour l'ensemble des clients")
        st.write('Les variables en rose ont contribué à accorder le crédit (donc à augmenter le score).\n Les variables en bleu ont contribué à refuser le crédit (donc à diminuer le score)')
        st_shap(shap.summary_plot(shap_values, df_shap))
    

    with tab3:
        st.header("Comparaison aux autres clients")
        categ = ['CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation']
        
        col1, col2 = st.columns(2)
        with col1:
            liste_variables1 = ['Revenus par personne', 'ID_Client', 'Genre', 'Âge', 'Ancienneté de l\'emploi', 'Revenus totaux',
                                'Nombre de personnes dans la famille', 'Voiture personnelle', 'Education secondaire',
                                'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit',
                                'Fréquence de paiement', 'Montant des annuités', 'Source externe 2', 'Source externe 3']
                    
            variable1 = st.selectbox("Sélectionnez la première variable à afficher", liste_variables1, key=1)
            var_en1 = get_english_var(variable1)
            if var_en1 in categ:
                var1_cat = 1
            else:
                var1_cat = 0
        
        with col2:
            liste_variables2 = ['Ancienneté de l\'emploi', 'ID_Client', 'GeFnre', 'Âge', 'Revenus totaux',
                                'Nombre de personnes dans la famille', 'Revenus par personne', 'Voiture personnelle',
                                'Education secondaire', 'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit',
                                'Fréquence de paiement', 'Montant des annuités', 'Source externe 2', 'Source externe 3']
            variable2 = st.selectbox("Sélectionnez la seconde variable à afficher", liste_variables2, key=2)
            var_en2 = get_english_var(variable2)
            if var_en2 in categ:
                var2_cat = 1
            else:
                var2_cat = 0
        
        df_comp = pd.read_csv("https://raw.githubusercontent.com/jlu0915/P7/master/Dashboard/df_comp_bis.csv")
        #df_comp_url = "https://raw.githubusercontent.com/jlu0915/P7/master/Dashboard/df_comp_bis.csv"
        #df_comp = pd.read_csv(df_comp_url)
        df_comp = feature_engineering(df_comp)
        new_df = feature_engineering(df_comp)
        if variable1 == variable2:
            df_comp = df_comp[[var_en1, 'TARGET', 'Score']].dropna()
        else:   
            df_comp = df_comp[[var_en1, var_en2, 'TARGET', 'Score']].dropna()
        
        col1_, col2_ = st.columns(2)
        with col1_:
            if var1_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig1 = px.histogram(df_comp, x=var_en1, color='TARGET', marginal=marg, nbins=50)
            if var1_cat == 0:
                fig1.add_vline(x=new_df[var_en1].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(option))
            fig1.update_layout(barmode='overlay', title={'text': variable1, 'x': 0.5, 'xanchor': 'center'})
            fig1.update_traces(opacity=0.75)
            st.plotly_chart(fig1, use_container_width=True)
        with col2_:
            if var2_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig2 = px.histogram(df_comp, x=var_en2, color='TARGET', marginal=marg, nbins=50)
            if var2_cat == 0:
                fig2.add_vline(x=new_df[var_en2].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(option))
            fig2.update_layout(barmode='overlay', title={'text': variable2, 'x': 0.5, 'xanchor': 'center'})
            fig2.update_traces(opacity=0.75)
            st.plotly_chart(fig2, use_container_width=True)
        
        if ((var1_cat + var2_cat) == 0) or (var1_cat == 1 and var2_cat == 0):
            scat = px.scatter(df_comp, x=var_en2, y=var_en1, color='Score', opacity=0.75,
                              color_continuous_scale=[(0.0, 'darkred'), (0.5, 'red'),
                                                      (0.5, 'red'), (0.7, 'orange'),
                                                      (0.7, 'orange'), (0.91, 'yellow'),
                                                      (0.91, 'green'),  (1.0, 'green')])
            scat.add_trace(go.Scatter(x=new_df[var_en2], y=new_df[var_en1], mode='markers',
                                      marker=dict(size=16, color='#8f00ff'), opacity=0.99, name='Client ' + str(option)))
            scat.update_layout(legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(scat, use_container_width=True)
        elif (var1_cat + var2_cat) == 2:
            table = np.round(pd.pivot_table(df_comp, values='Score', index=[var_en1],
                                            columns=[var_en2], aggfunc=np.mean),
                             2) 
            fig = px.imshow(table, text_auto=True, color_continuous_scale='Blues')
            #[(0.0, 'red'), (0.5, 'orange'), (0.5, 'orange'), (0.7, 'yellow'), (0.7, 'yellow'), (0.91, 'lime'), (0.91, 'lime'),  (1.0, 'green')]
            st.write(fig)
        else:
            scat = px.scatter(df_comp, x=var_en1, y=var_en2, color='Score', opacity=0.75,
            color_continuous_scale=[(0.0, 'darkred'), (0.5, 'red'),
                                    (0.5, 'red'), (0.7, 'orange'),
                                    (0.7, 'orange'), (0.91, 'yellow'),
                                    (0.91, 'green'),  (1.0, 'green')])
            scat.add_trace(go.Scatter(x=new_df[var_en1], y=new_df[var_en2], mode='markers',
                                      marker=dict(size=16, color='#8f00ff'), opacity=0.99, name='Client ' + str(option)))
            scat.update_layout(legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))

with st.sidebar:
    st.image('logo.png')
    option = st.selectbox(
        'Client ID',
        df_dashboard['SK_ID_CURR'].unique())
    if option:
        flag = df_dashboard[df_dashboard["SK_ID_CURR"] == option]
        flag1 = df[df["SK_ID_CURR"] == option]
        col3, col4, col5 = st.columns([2, 8, 2])
        with col4:
            st.subheader('General Information')
        st.write('Genre：{}'.format(flag1['CODE_GENDER'].apply(lambda x: 'Woman' if x == 1 else 'Men').values[0]))
        st.write('Age：{}'.format(flag1['DAYS_BIRTH'].apply(lambda x: round(-x / 365.25, 0)).values[0]))
        st.write('Revenus totaux：{} k'.format(flag1['AMT_INCOME_TOTAL'].values[0] / 1000))
        st.write('Anciennete emploi：{} year'.format(round(-flag1["DAYS_EMPLOYED"].values[0] / 365.25, 1)))

predict()
#if __name__ == "__predict__":
    #predict()
