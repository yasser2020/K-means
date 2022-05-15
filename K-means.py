# -*- coding: utf-8 -*-
"""
Created on Sun May 15 00:53:33 2022

@author: Yasser Ezzat
"""

import pandas as pd

df=pd.read_excel('data/K_Means.xlsx')
print(df.head())

import seaborn as sns

# sns.regplot(x=df['X'],y=df['Y'],fit_reg=False)
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
model=kmeans.fit(df)
predicated_values=model.predict(df)

from matplotlib import pyplot as plt

plt.scatter(df['X'],df['Y'],c=predicated_values,s=50,cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='red',alpha=0.5)



