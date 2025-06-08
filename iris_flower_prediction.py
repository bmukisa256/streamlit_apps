import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
         # Iris Flower Prediction App 
         
         This app predicts the species of ***iris flowers*** based on their features.
         """)

st.sidebar.header('User Input Features')

def user_input_features():
    """
    Function to get user input for the iris flower features.
    Returns a DataFrame with the input values.
    """
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 5.0, 3.0)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Features')
st.write(df)

iris = datasets.load_iris()
X= iris.data
y = iris.target

st.subheader('Using a Random Forest Classifier')
clf = RandomForestClassifier()
clf.fit(X, y)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Model Accuracy')
st.write(clf.score(X, y))       
