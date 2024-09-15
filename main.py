import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("/content/titanic_train.csv")
df.head()
df.shape
df.columns
df.duplicated().sum()
df.isnull().sum()
df = df.drop('Cabin', axis = 1)
df = df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
df
object_col = df.select_dtypes(include = 'object').columns.tolist()
numeric_col = df.select_dtypes(include = ['int', 'float']).columns.tolist()
print("object columns : ", object_col)
print("numeric columns : ", numeric_col)

#the numerical data is filled with the mean
for feature in numeric_col:
  df[feature].fillna(df[feature].mean(), inplace = True)

#where the categorical data will be filled with mode
for feature in object_col:
  df[feature].fillna(df[feature].mode()[0], inplace = True)
df.info()
df.describe()
df.nunique()


  
