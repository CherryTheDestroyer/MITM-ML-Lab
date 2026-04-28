# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns

#Simulate Data
np.random.seed(42)
n_samples=1000

#Simulated features
age=np.random.randint(18,70,n_samples)
income=np.random.randint(2000,15000,n_samples)
browsing_time=np.random.normal(5,2,n_samples)
clicked=(0.3*(age<30)+0.4*(income<50000)+0.5*(browsing_time>5)+np.random.normal(0,0.2,n_samples))>0.8
clicked=clicked.astype(int)

#Create Dataframe
df=pd.DataFrame({'Age':age,'Annual_Income':income,'Browsing_Hours':browsing_time,'Clicked_on_Ad':clicked})

#Split data
X=df[['Age','Annual_Income','Browsing_Hours']]
y=df['Clicked_on_Ad']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Logistic Regression Model
model=LogisticRegression()
model.fit(X_train,y_train)

#Prediction and Evaluation
y_pred=model.predict(X_test)
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print(df)

#Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()