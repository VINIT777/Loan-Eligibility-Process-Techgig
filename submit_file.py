import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('test_data.csv')

from scipy.stats import mode

df['Gender'].fillna(mode(df['Gender'].dropna()).mode[0], inplace=True)
df['Self_Employed'].fillna(mode(df['Self_Employed'].dropna()).mode[0], inplace=True)
df['Married'].fillna(mode(df['Married'].dropna()).mode[0], inplace=True)
df['Dependents'].fillna(mode(df['Dependents'].dropna()).mode[0], inplace=True)
# Numeric Variable
df['Credit_History'].fillna(mode(df['Credit_History']).mode[0], inplace=True)
df['Loan_Amount_Term'].fillna(mode(df['Loan_Amount_Term']).mode[0], inplace=True) #revaluate with mean
df['LoanAmount'].fillna(np.mean(df['LoanAmount']), inplace=True)

mv = df.apply(lambda x: sum(x.isnull()),axis=0)

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes

df['emi'] = df['LoanAmount']*1000/df['Loan_Amount_Term']
df['idu'] = df['ApplicantIncome']/(df['Dependents'] + 1) 
df['Ratio'] = df['idu']/df['emi']

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
rge = (max(df['Ratio']) - min(df['Ratio']))
df['Ratio'] = (df['Ratio'] - min(df['Ratio']))/rge
  
fe = ['Ratio','Credit_History','Married','Education','Self_Employed','Property_Area']  
fet = df[fe]
des = df[fe].describe()


y_pred = RF.predict(fet)

df['y_pred'] = y_pred
sub = pd.DataFrame()
sub['Application_ID'] = df['Application_ID']
sub['Loan_Status'] = df['y_pred'].replace([1, 0],['Y', 'N'])
sub.to_csv('submission4.csv', index=False)