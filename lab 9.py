import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

#Load Dataset
# 1. Load dataset (after downloading and placing 'spam.csv' in your folder)
df=pd.read_csv(r"D:\College\6th sem\ML Lab\archive\spam.csv",
encoding='latin-1')
df.columns =['label', 'message']

#Encode labels
df['label']=df['label'].map({'ham':0,'spam':1})

#Text vectorization
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['message'])
y=df['label']

#Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Train Naive model
model=MultinomialNB()
model.fit(X_train,y_train)

#Predict
y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\n Classification Report:\n",classification_report(y_test,y_pred))
