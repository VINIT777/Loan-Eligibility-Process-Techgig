import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('train_data.csv')

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
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes

R = (9.5/(12*100))
df['TotalInc'] = df['ApplicantIncome']+df['CoapplicantIncome']

df['emi'] = ((df['LoanAmount']*1000)*R*((1+R)**df['Loan_Amount_Term']))/((1+R)**(df['Loan_Amount_Term'])+1)

df['idu'] = (df['TotalInc']/12)/(df['Dependents'] + 1)

df['Cred'] = df['ApplicantIncome']/(df['LoanAmount']*1000)
df['TCred'] = df['TotalInc']/(df['LoanAmount']*1000)

df['Ratio'] = (df['idu'])/df['emi']

# Feature Scaling
#from sklearn.preprocessing import MinMaxScaler

scale_var = ['Credit_History','Cred','TCred','emi','Ratio']
for i in scale_var:
    rge = (max(df[i]) - min(df[i]))
    df[i] = (df[i] - min(df[i]))/rge
df.dtypes

  
fe = ['Credit_History','emi','Ratio']#'Cred','TCred'] 
fet = df[fe]
des = df[fe].describe()

X = df[fe].values
Y = df['Loan_Status'].values
      
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, random_state = 0)

