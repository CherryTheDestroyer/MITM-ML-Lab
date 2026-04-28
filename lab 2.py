
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

iris= load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)

np.random.seed(0)
X.loc[X.sample(frac=0.1).index,'sepal length (cm)']= np.nan

X['flower_type']=np.random.choice(['Type A','Type B','Type C'], size=len(X))

imputer = SimpleImputer(strategy='mean')
X_numeric_imputed=imputer.fit_transform(X[iris.feature_names])
X_numeric_imputed_df=pd.DataFrame(X_numeric_imputed, columns=iris.feature_names)

print("After Imputation (Missing Values Handled):")
print(X_numeric_imputed_df.head())

encoder = OneHotEncoder(handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X[['flower_type']])
encoded_df=pd.DataFrame(X_categorical_encoded.toarray(),
columns=encoder.get_feature_names_out())
print("\n After One-Hot Encoding (Categorical Data Handled):")
print(encoded_df.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric_imputed_df)
X_scaled_df=pd.DataFrame(X_scaled, columns=iris.feature_names)

print("\n After Standardization (Feature Scaling Applied):")
print(X_scaled_df.head())

X_processed=pd.concat([X_scaled_df,encoded_df],axis=1)

print("\n Final Preprocessed Dataset:")
print(X_processed.head())
print("\n Final shape:", X_processed.shape)