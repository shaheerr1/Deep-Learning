import pandas as pd
import numpy as np
import gradio as gr

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest


df = pd.read_csv('C:/datasets/heart.csv')


# -----------------------------------> Understanding the dataset
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.describe(include=['O']))


# -----------------------------------> EDA
df.hist(bins=30, figsize=(15, 10), grid=False)
plt.show()

# ----------------------------------->
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
plt.show()


# ----------------------------------->
sns.pairplot(df, hue='target', vars=[
             'age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
plt.show()


# ----------------------------------->
for column in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[column])
    plt.title(f'Box plot of {column}')
    plt.show()


# ----------------------------------->
# Analyzing the 'target' variable
print(df['target'].value_counts())
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
plt.title('Heart Disease (Target) Frequency')
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Count')
plt.show()


# ----------------------------------->
X = df.drop('target', axis=1)
y = df['target']


# -----------------------------------> Feature selection
k = 10
selector = SelectKBest(score_func=chi2, k=k)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support(indices=True)]
print("Selected Features: ", selected_features)


# -----------------------------------> Spliting and scaling the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------> Using Naive Bayes classifier
nb_classifier = GaussianNB()


# -----------------------------------> Training
nb_classifier.fit(X_train_scaled, y_train)


# -----------------------------------> GUI
def predict_heart_disease(*args):
    new_data = [float(i) for i in args]
    new_data_array = np.array(new_data).reshape(1, -1)
    new_data_scaled = scaler.transform(new_data_array)
    new_prediction = nb_classifier.predict(new_data_scaled)
    return "Positive" if new_prediction[0] == 1 else "Negative"


input_fields = [gr.components.Number(label=feature)
                for feature in selected_features]


interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=input_fields,
    outputs="text",
    title="Heart Disease Prediction",
    description="Input the required fields and click submit to predict the likelihood of heart disease.",
    theme="huggingface"
)
interface.launch()
