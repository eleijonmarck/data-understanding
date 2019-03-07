import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# food prefreences,

# len 10
dishes = ["pad thai", "pizza", "pancakes", "kebab", "dumplings", "sushi", "hot dog", "poke bowl", "noodles", "hamburger"]

persons = [
        [1,0,0,0,1,1,0,0,1,0], # Asian
        [1,1,0,1,0,0,1,0,0,1], # Fast food
        [1,1,0,0,0,0,1,0,0,1], # Fast food
        [0,1,0,0,0,0,1,0,0,1], # Fast Food
        [0,0,0,0,0,1,0,1,0,0], # Asian
        [1,0,0,0,0,1,0,1,0,1], # Asian
        [0,0,1,0,0,1,0,1,0,0], # Asian
        [0,0,1,0,0,0,1,0,0,0], # US
        [0,0,1,0,0,0,1,0,0,1], # US
        [0,1,0,1,0,0,0,0,0,1], # US
        [0,1,1,1,0,0,0,0,0,1], # US
        [0,1,0,1,0,1,0,0,0,1], # US
        [0,0,0,0,0,0,1,0,1,0], # Asian
        [0,1,0,0,0,0,1,0,0,0], # US
        [0,0,0,0,1,1,0,1,0,0], # Asian
        ]

df = pd.DataFrame(persons, columns=dishes)

pca = PCA(n_components=2)
pca.fit(df)

print(pca.components_)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

import matplotlib.pyplot as plt

pca_df = pd.DataFrame(pca.components_, columns=dishes)


top = pca_df.loc[0].reindex(pca_df.loc[0].abs().nlargest(10).sort_values(ascending=False).index)

top.plot(kind='bar')
plt.show()


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');
