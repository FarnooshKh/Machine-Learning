import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# save load_iris() sklearn dataset to iris
iris = load_iris()

# np.c_ h is used to concat iris['data'] and iris['target'] arrays
# for pandas column argument: concat iris['feature_names'] list
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
data.to_csv('Data/iris.csv')

