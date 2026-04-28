import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species']=iris.target
df['species']=df['species'].map({i:species for i,species in enumerate(iris.target_names)})

print("First 5 rows of dataset:")
display(df.head())

print("/n Dataset info")
df.info()
print("/n Summary Statistics:")
display(df.describe())

sns.pairplot(df,hue="species")
plt.suptitle("Pairplot of iris feature by species",y=1.02)
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:,:-1].corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

for feature in iris.feature_names:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='species',y=feature,data=df)
    plt.title(f"{feature.capitalize()}by species")
    plt.show()
