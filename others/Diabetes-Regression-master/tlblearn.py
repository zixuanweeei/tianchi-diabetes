import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def accu(y_true, y_pre):
    m = len(y_true)
    sum = 0.0
    for i , j in zip(y_true, y_pre):
        t = j - i
        sum += t * t
    return sum / (2 * m)


## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./data/d_train_20180102.csv', encoding="utf-8")
df = df.replace({'男','女'},{np.float64(1).item(),np.float64(0).item()})
df = df.fillna(df.mean())
Y = df['血糖']
X = df.drop(['id', '体检日期', '血糖'], 1, inplace=False)
# print(x_test.shape)

# 线性多项式扩展提高准确率
models = [
    Pipeline([
        ('pca', PCA(n_components=3)),  # 进行降维操作，降低成为3个维度
        #('Poly', PolynomialFeatures()),  # 给定进行多项式扩展操作
        #('norm', Normalizer()),  # 进行归一化操作
         #('Linear', DecisionTreeRegressor(criterion='mse'))  # 回归模型
        ('Linear', Lasso())
    ])
]
model = models[0]
N = 10000
d_pool = np.arange(1, N, 1)  # 阶
for i in d_pool:
#model.set_params(Poly__degree=d)  ## 设置多项式的阶乘
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print(x_train.head())
    model.fit(x_train, y_train)
    lin = model.get_params('Linear')['Linear']
    y_hat = model.predict(x_test)
    s = accu(y_test, y_hat)
    print(u'均方误差=%.3f' % (s))
    if(s > 0.66):
        testdf = pd.read_csv('./data/d_test_A_20180102.csv', encoding="utf-8")
        testdf = testdf.fillna(df.mean())
        testdf = testdf.replace({'男','女'},{1.0,0.0})
        testX = testdf.drop(['id',  '体检日期'], 1, inplace=False)
        pred_Y = model.predict(testX)
        print(pred_Y)
        # 行转列导出csv
        adfY = pred_Y.reshape(-1, len(pred_Y)).T
        pd.set_option('precision', 3)
        dfY = pd.DataFrame(adfY)
        filename= './result.csv'
        dfY.to_csv(filename)
        break

