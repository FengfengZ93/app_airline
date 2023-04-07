import streamlit as st
import pandas as pd
from index import load_data
from joblib import load
from sklearn.inspection import partial_dependence
import plotly.express as px
import matplotlib.pyplot as plt


df = load_data()
model = load ('best_model_rf.joblib')

rf_model = model.named_steps['model']

importances = rf_model.feature_importances_

preprocessor = model.named_steps['preprocessor']
cat_transformer = preprocessor.named_transformers_['cat']
cat_feature_names = cat_transformer.get_feature_names_out(['Gender', 'Customer Type','Type of Travel','Class'])
feature_names = list(cat_feature_names) + ['Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
                               'Food and drink','Gate location', 'Inflight wifi service', 'Inflight entertainment', 
                               'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 
                               'Baggage handling', 'Checkin service']

feature_importances = dict(zip(feature_names, importances))

sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
sorted_features = [x[0] for x in sorted_importances]
sorted_importances = [x[1] for x in sorted_importances]


feature_importance_df = pd.DataFrame({'feature': sorted_features,
                                      'importance': sorted_importances})

st.dataframe(feature_importance_df)

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(sorted_features, sorted_importances)
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Feature Importances (Random Forest)')

st.pyplot(fig)


var = st.selectbox('Select the values:', sorted_features)

pd_values = partial_dependence(model, df, var, kind="average")
pd_df =  pd.DataFrame({var: pd_values['values'][0], 
                       'average_prediction': pd_values['average'][0]
                       })
fig = px.line(pd_df, x= var, y='average_prediction')
st.plotly_chart(fig)
