import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
print(train.head())

# shape
print('shape:',train.shape)
print('test_shape:',test.shape)

# analysis saleprice: there is no null values and there are all number values

print(train['SalePrice'].describe())
# heatmap for price. lighter color is higher relationship
cor=train.corr()
# print(cor)
# f,ax=plt.subplots(figsize=(15,9))
# sns.heatmap(cor,vmax=0.8,square=True)
# plt.show()
# top10 relationship
# k=10
# cols=cor.nlargest(k,'SalePrice')['SalePrice'].index
# cm=np.corrcoef(train[cols].values.T)
# sns.set(font_scale=1.25)
# hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},\
#                yticklabels=cols.values,xticklabels=cols.values)
# plt.show()

# pairplot
sns.set()
relcols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','TotRmsAbvGrd','YearBuilt']
sns.pairplot(train[relcols],size=1.2)
plt.show()






# sns.distplot(train['SalePrice'])
# plt.show()

# Air and Price
# sns.barplot(x='CentralAir',y='SalePrice',data=train)
# plt.show()

# OverallQual and price
# sns.barplot(x='OverallQual',y='SalePrice',data=train)
# plt.show()

# YearBuild and price
# sns.scatterplot(x='YearBuilt',y='SalePrice',data=train)
# plt.show()


# LotArea
# sns.scatterplot(x='LotArea',y='SalePrice',data=train)
# plt.show()
# train.head()
