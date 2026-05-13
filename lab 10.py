import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Simulate customer RFM data
np.random.seed(42)
n_customers=200
data=pd.DataFrame({'CustomerID':range(1,n_customers+1),
                   'Recency':np.random.randint(1,100,size=n_customers),
                   'Frequency':np.random.randint(1,20,size=n_customers),
                   'Monetary':np.random.randint(100,10000,size=n_customers)})

#Standardize RFM features
scaler=StandardScaler()
rfm_scaled=scaler.fit_transform(data[['Recency','Frequency','Monetary']])

#Find optimal number of clusters
sse=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)
    
#Plot Elbow curve
plt.figure(figsize=(6,4))
plt.plot(range(1,11),sse,marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters")
plt.ylabel("SSE(Inertia)")
plt.grid(True)
plt.show

#Apply K-Means
kmeans=KMeans(n_clusters=4,random_state=42)
data['Cluster']=kmeans.fit_predict(rfm_scaled)

#Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=data,x='Recency',y='Monetary',hue='Cluster',palette='Set2')
plt.title("Customer Segmentation(Recency vs Monetary)")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=data,x='Frequency',y='Monetary',hue='Cluster',palette='Set1')
plt.title("Customer Segmentation(Frequency vs Monetary)")
plt.show()

#Cluster summary
cluster_summary=data.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
print("\n Cluster Summary:\n",cluster_summary)
