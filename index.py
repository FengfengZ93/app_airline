import streamlit as st
import pandas as pd

st.title('Welcome to airline sentiment analysis!')

def load_data():
    url = 'https://raw.githubusercontent.com/remijul/dataset/master/Airline%20Passenger%20Satisfaction.csv'
    df = pd.read_csv(url, sep=';')
    df.dropna(inplace=True) 
    return df

df_load = load_data()
st.dataframe(df_load.head(100))

