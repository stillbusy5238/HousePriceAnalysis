import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as np


train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
print(train.head())

# shape
print('shape:',train.shape)
print('test_shape:',test.shape)

# analysis saleprice: there is no null values and there are all number values

print(train['SalePrice'].describe())


# train.head()
