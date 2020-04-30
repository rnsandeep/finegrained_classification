import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import pickle


cls_dict = pickle.load(open('cls_dict.pkl', 'rb'))

label_dict = pickle.load(open('labels.pkl', 'rb'))
upc_brand = pickle.load(open('upc_brand.pkl','rb'))

features = []
labels = []
for image in cls_dict:
    feature = cls_dict[image].cpu().detach().numpy().reshape((1000, ))
    features.append(feature)
    labels.append(upc_brand[image.split('/')[-2]])
    #labels.append(label_dict[image.split('/')[-1]+'\n']) # upc labels

X = np.array(features)  #np.random.rand(10000,1000) #np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array(labels) #np.random.randint(10, size=10000)

#pca = PCA(n_components=50)
#pca_result = pca.fit_transform(X)


#print(pca_result)
#print(pca.singular_values_)
#PCA(n_components=2)

df_subset = pd.DataFrame()

#df_subset['pca-one'] = pca_result[:,0]
#df_subset['pca-two'] = pca_result[:,1] 
#df_subset['pca-three'] = pca_result[:,2]

df_subset['y'] = y

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 258),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.savefig('plot.png')
