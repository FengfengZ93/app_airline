import streamlit as st
import pandas as pd
import datetime
from index import load_data
import plotly.express as px
import plotly.io as pio
import plotly.subplots as sp
import plotly.graph_objs as go

df = load_data()

columns = ['Class', 'Customer Type', 'Gender', 'Satisfaction']

traces = []
for column in columns:
    grouped = df.groupby(column)[column].count()
    traces.append(go.Pie(values=grouped.values, labels=grouped.index, name=column))

fig = sp.make_subplots(rows=1, cols=len(columns), specs=[[{'type':'domain'}]*len(columns)])
for i, trace in enumerate(traces):
    fig.add_trace(trace, 1, i+1)

fig.update_layout(title='Pie Charts')
st.plotly_chart(fig)

fig = px.scatter(df, x="Arrival Delay in Minutes", y="Departure Delay in Minutes")
st.plotly_chart(fig)
