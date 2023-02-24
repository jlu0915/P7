import streamlit as st
import pandas as pd
import pickle
import warnings
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
'''
st.set_page_config(page_title='Loan Scoring APP', layout="wide")

df_test = pd.read_csv('df_API.csv')
model = pickle.load(open('LGBM.pickle', 'rb')).best_estimator_

st.markdown(sysmenu, unsafe_allow_html=True)


def predict():
    col1, col2 = st.columns([3.5, 6.5])
    with col2:
        st.title('_solvency analysis_')
    flag.drop(['SK_ID_CURR'], axis=1, inplace=True)
    probability_default_payment = model.predict_proba(flag)[:, 1]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability_default_payment[0] * 100,
        title={'text': "Client score"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(yaxis={'range': [0, 100]})

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(flag)
    shap.summary_plot(shap_values, flag, show=False)
    plt.savefig('features importance.png', dpi=100)

    tab1, tab2 = st.tabs(["Client score", "Importance features"])
    with tab1:
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.image('features importance.png')
    if probability_default_payment[0] >= 0.92:
        prediction = "Prêt NON Accordé"
    else:
        prediction = "Prêt Accordé"
    with st.sidebar:
        st.write('Predict：{}'.format(prediction))


with st.sidebar:
    st.image('logo.png')
    option = st.selectbox(
        'Client ID',
        df_test['SK_ID_CURR'].unique())
    if option:
        flag = df_test[df_test["SK_ID_CURR"] == option]
        col3, col4, col5 = st.columns([2, 8, 2])
        with col4:
            st.subheader('General Information')
        st.write('Gender：{}'.format(flag['CODE_GENDER'].apply(lambda x: 'Woman' if x == 1 else 'Men').values[0]))
        st.write('Age：{}'.format(flag['DAYS_BIRTH'].apply(lambda x: round(-x / 365.25, 0)).values[0]))
        st.write('Total revenue：{} k'.format(flag['AMT_INCOME_TOTAL'].values[0] / 1000))
        st.write('Seniority：{} year'.format(round(-flag["DAYS_EMPLOYED"].values[0] / 365.25, 1)))
predict()
