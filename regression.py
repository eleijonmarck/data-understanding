import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv('current_processed.csv')

print(df.shape)

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

model = LinearRegression()
pca = PCA(n_components=2)

pca.fit(df)


print(pca.components_)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

import matplotlib.pyplot as plt

pca_df = pd.DataFrame(pca.components_, columns=df.columns)

top_pca = pca_df.reindex(pca_df[0].abs().sort_values().index)
pca_df.plot(kind='bar')
plt.show()
