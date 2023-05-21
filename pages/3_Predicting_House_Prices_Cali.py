import joblib
import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os

st.set_page_config(
    page_title="Dự đoán giá nhà ở California",
    page_icon="💰"
)

css = """
    <style>
        .css-6qob1r {
            background-color: #e88102;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

def my_format(x):
    s = "{:,.0f}".format(x)
    L = len(s)
    if L < 14:
        s = '&nbsp'*(14-L) + s
    return s

url = 'https://drive.google.com/uc?id=1_8_ZBrXtkv08oXteYW4z3X-Q_cvDplxr'
output = './src/Predicting_House_Prices_Cali/forest_reg_model.pkl'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

with open(output, 'rb') as f:
    forest_reg = joblib.load(f)

column_names=['longitude','latitude','housing_median_age','total_rooms',
              'total_bedrooms','population','households','median_income',
              'rooms_per_household','population_per_household',
              'bedrooms_per_room','ocean_proximity_1', 
              'ocean_proximity_2', 'ocean_proximity_3', 
              'ocean_proximity_4', 'ocean_proximity_5']
st.title('Dự báo giá nhà California')
x_test = pd.read_csv('./src/Predicting_House_Prices_Cali/x_test.csv', header = None, names=column_names)
y_test = pd.read_csv('./src/Predicting_House_Prices_Cali/y_test.csv', header = None)
y_test = y_test.to_numpy()
N = len(x_test)
st.dataframe(x_test)
get_5_rows = st.button('Lấy 5 hàng ngẫu nhiên và dự báo')
if get_5_rows:
    index = np.random.randint(0,N-1,5)
    some_data = x_test.iloc[index]
    st.dataframe(some_data)
    result = 'y_test:' + '&nbsp&nbsp&nbsp&nbsp' 
    for i in index:
        s = my_format(y_test[i,0])
        result = result + s
    result = '<p style="font-family:Consolas; color:Blue; font-size: 15px;">' + result + '</p>'
    st.markdown(result, unsafe_allow_html=True)

    some_data = some_data.to_numpy()
    y_pred = forest_reg.predict(some_data)
    result = 'y_predict:' + '&nbsp'
    for i in range(0, 5):
        s = my_format(y_pred[i])
        result = result + s
    result = '<p style="font-family:Consolas; color:Blue; font-size: 15px;">' + result + '</p>'
    st.markdown(result, unsafe_allow_html=True)

