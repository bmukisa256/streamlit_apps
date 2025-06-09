import streamlit as st 
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set the title of the app
st.title('Multi Model App')
st.write("""
            This app allows you to select and run different machine learning classifiers.
            """)

dataname = st.sidebar.selectbox(
    'Select Dataset',('Iris', 'Wine dataset', 'Breast Cancer'))

classifier = st.sidebar.selectbox(
    'Select Classifier', ('KNN', 'SVM', 'Decision Tree', 'Random Forest'))

# Load dataset based on user selection
@st.cache_data
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine dataset':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    # Extract features and target
    # data.data contains the features, data.target contains the labels
    X = data.data
    y = data.target
    return X, y

# Get the dataset
X,y = get_dataset(dataname)

# Display dataset information
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))


# Add UI for classifier parameters
# This function will create sliders or input fields for the parameters of the selected classifier
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K

    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C

    elif clf_name == 'Decision Tree':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth

    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

# Get classifier parameters based on user selection
# This function will call add_parameter_ui to create the UI for the selected classifier
params = add_parameter_ui(classifier)

# Get the classifier based on user selection and parameters
# This function will return the classifier object based on the selected classifier and its parameters
def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])

    elif clf_name == 'SVM':
        clf = SVC(C=params['C'])

    elif clf_name == 'Decision Tree':        
        clf = DecisionTreeClassifier(max_depth=params['max_depth'])

    else:
        clf = RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'])
    return clf

# Get the classifier object based on user selection and parameters
# This will create the classifier object that will be used for training and prediction
clf = get_classifier(classifier, params)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Display the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the results in the Streamlit app
st.subheader('Classifier Performance')
st.write(f'Dataset: {dataname}')
st.write(f'Classifier: {classifier}')
st.write(f'Accuracy: {accuracy:.2f}')


# Display the predictions
st.subheader('Predictions')
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the predictions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

x1= X_pca[:, 0]
x2 = X_pca[:, 1]

fig = plt.figure(figsize=(10, 6))
sns.scatterplot(x=x1, y=x2, hue=y, palette='viridis', s=100)
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
st.pyplot(fig)
