import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import ssl
ssl._create_default_https_context=ssl._create_unverified_context

#
#FUNCTION TO APPLY PCA AND PLOT
#
def apply_pca_and_plot(X, y, title, cmap):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca=pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1],c=y, cmap=cmap, s=10)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()

#PART A - MNIST (IMAGE DATA)
#
print("Loading MNIST dataset ... ")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_mnist = mnist.data
y_mnist = mnist.target.astype(int)

apply_pca_and_plot(
X_mnist,
y_mnist,
"PCA Visualization of MNIST Dataset",
"tab10"

)

#
#PART B -20 NEWSGROUPS (TEXT DATA)
#
print("Loading 20 Newsgroups dataset ... ")
news = fetch_20newsgroups(subset='train',
remove=('headers', 'footers', 'quotes'))

vectorizer= TfidfVectorizer(max_features=2000, stop_words='english')
X_news = vectorizer.fit_transform(news.data).toarray()
y_news = news.target

apply_pca_and_plot(X_news,y_news,"PCA Visualization of 20 Newsgroups Dataset","tab20")


