import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split


train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
# print(train.head())

# shape
# print('shape:',train.shape)
# print('test_shape:',test.shape)

# analysis saleprice: there is no null values and there are all number values

# print(train['SalePrice'].describe())
# heatmap for price. lighter color is higher relationship
# cor=train.corr()
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
# sns.set()
# relcols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','TotRmsAbvGrd','YearBuilt']
# sns.pairplot(train[relcols],size=1.2)
# plt.show()
# no null values data
# print(train[relcols].isnull().sum())
train_cols=['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','TotRmsAbvGrd','YearBuilt']
features=train[train_cols].values
labels=train['SalePrice'].values
# standarScaler data
# features_scaled=preprocessing.StandardScaler().fit_transform(features)
# print(features_scaled)
#
# labels_scaled=preprocessing.StandardScaler().fit_transform(labels.reshape(-1,1))

X_train,X_val,Y_train,Y_val=train_test_split(features,labels,test_size=0.33,random_state=42)

# model
model=RandomForestRegressor(n_estimators=400)
model.fit(X_train,Y_train)

Y_pred=model.predict(X_val)
# print(Y_pred)
# print(Y_val)
# print(sum(abs(Y_pred-Y_val)/len(Y_pred)))
#
# print(test[train_cols].isnull().sum())
print(test['GarageCars'].median())
print(test['TotalBsmtSF'].median())
test['GarageCars'].fillna(2,inplace=True)
test['TotalBsmtSF'].fillna(988,inplace=True)
print(test[train_cols].isnull().sum())
x=test[train_cols].values
Y_test_pred=model.predict(x)
print(Y_test_pred)
prediction=pd.DataFrame(Y_test_pred,columns=['SalePrice'])
result=pd.concat([test['Id'],prediction],axis=1)
print(result.columns)
result.to_csv('./Submissions_1.csv',index=False)










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
