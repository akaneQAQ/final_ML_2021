import copy
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

path = 'C:/Users/shq20/Desktop/dataset_finalproject'
df = pd.read_csv(path + '/vaers_preprocessed_2.csv')

np.set_printoptions(precision=6, suppress=True)
X = np.asarray(df[['MODERNA', 'PFIZER\BIONTECH', 'AGE_YRS', 'SEX', 'DISABLE',
                   'OTHER_MEDS', 'CUR_ILL', 'asthma', 'hypertension', 'diabete', 'allergy',
                   'anxiety', 'depression', 'thyroid', 'hyperlipidemia', 'heart', 'pain',
                   'cancer', 'obesity', 'migraines', 'covid', 'arthritis', 'kidney']])
Y = np.asarray(df['NUMDAYS'], dtype=np.float)

## 探索性数据分析 EDA
# 画年龄、天数的直方图
plt.axes(yscale="log")
plt.hist(Y, color='slateblue', edgecolor='k', alpha=0.6)
plt.xlabel('NUMDAYS')
plt.ylabel('COUNT')
plt.show()
# 画16个病history有多少个人得过病
f = np.sum(X, axis=0)
plt.figure(figsize=(20, 10))
plt.xticks(range(16), ['asthma', 'hypertension', 'diabete', 'allergy',
                       'anxiety', 'depression', 'thyroid', 'hyperlipidemia', 'heart', 'pain',
                       'cancer', 'obesity', 'migraines', 'covid', 'arthritis', 'kidney'],
           size=11)
plt.bar(range(16), f[7:], color='lightblue', edgecolor='k', alpha=0.6, width=1)
plt.ylabel('COUNT')
plt.show()
# 其他特征
corr = df[['MODERNA', 'PFIZER\BIONTECH', 'AGE_YRS', 'SEX', 'DISABLE', 'OTHER_MEDS', 'CUR_ILL']].corr()
sns.set()
fig = plt.figure(figsize=(13, 8))
sns.heatmap(corr, annot=True, cmap="Purples")
plt.show()


def split_train_test(x, y, test_ratio=0.2):
    y = y.reshape(-1, 1)
    n = len(x)
    num_train = int((1 - n) * test_ratio)
    data = np.hstack((x, y))
    random.seed(100)
    random.shuffle(data)
    return data[:num_train, :X.shape[1]], data[:num_train, -1], data[num_train:, :X.shape[1]], data[num_train:, -1]


X_train, Y_train, X_test, Y_test = split_train_test(X, Y, test_ratio=0.2)

## 线性回归 OLS
result_OLS = np.dot(np.dot(np.linalg.inv(X_train.T @ X_train), X_train.T), Y_train)
MSE = 0
for i in range(len(X_train)):
    yiHat = np.sum(result_OLS * X_train[i])
    MSE += ((yiHat - Y_train[i]) ** 2) / len(X_train)
print('Training rmse:', MSE ** 0.5)
MSE = 0
for i in range(len(X_test)):
    yiHat = np.sum(result_OLS * X_test[i])
    MSE += ((yiHat - Y_test[i]) ** 2) / len(X_test)
print('Testing rmse:', MSE ** 0.5)
print('result_OLS', result_OLS)
plt.bar(range(len(result_OLS)), result_OLS)
plt.title('OLS')
plt.show()

## 带惩罚项的线性回归（ridge）
n = X_train.shape[0]
p = X_train.shape[1]
mse_ridge = []
result = []
alpha = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
for lamdda in alpha:
    XtX = X_train.T @ X_train
    R = np.linalg.inv(XtX + lamdda * np.eye(p))
    coef = R @ X_train.T @ Y_train
    MSE = 0
    for i in range(len(X_test)):
        yiHat = np.sum(coef * X_test[i])
        MSE += ((yiHat - Y_test[i]) ** 2) / len(X_test)
    mse_ridge.append(MSE)
    result.append(coef)
print('lambda:', alpha[np.argmin(mse_ridge)])
result_RIDGE = result[np.argmin(mse_ridge)]
MSE = 0
for i in range(len(X_train)):
    yiHat = np.sum(result_RIDGE * X_train[i])
    MSE += ((yiHat - Y_train[i]) ** 2) / len(X_train)
print('RIDGE Training rmse:', MSE ** 0.5)
print('Testing RMSE:', np.min(mse_ridge) ** 0.5)
print('result_RIDGE:', result_RIDGE)
plt.bar(range(len(result_RIDGE)), result_RIDGE)
plt.title('RIDGE')
plt.show()


## 带惩罚项的线性回归（lasso）
def k_fold_split(origin_X, origin_Y, m, k=5):  # 为交叉验证划分测试和验证集
    p = len(origin_X[0])
    N = len(origin_X)
    n = N // k

    train_X = np.zeros((N - n, p))
    train_Y = np.zeros((N - n, 1))
    valid_X = np.zeros((n, p))
    valid_Y = np.zeros((n, 1))

    train_X[:m * n] = origin_X[:m * n]
    train_X[m * n:] = origin_X[(m + 1) * n:]
    train_Y[:m * n] = origin_Y[:m * n]
    train_Y[m * n:] = origin_Y[(m + 1) * n:]

    valid_X = origin_X[m * n:(m + 1) * n]
    valid_Y = origin_Y[m * n:(m + 1) * n]

    return train_X, train_Y, valid_X, valid_Y, n


def convergence(a, b, c, epsilon):
    if (np.linalg.norm(a) + np.linalg.norm(b) + np.linalg.norm(c)) < epsilon:
        return True
    else:
        return False


def S(u, p):
    for i in range(len(u)):
        if abs(u[i]) > p:
            u[i] -= np.sign(u[i]) * p
        else:
            u[i] = 0
    return u


# proximal
def prox(x, y, lamdda, p, mu=1e-11, e=0.0001):
    Beta_t1 = np.random.random((p, 1))
    for a in range(50000):  # 迭代上限次数
        Beta_t = copy.deepcopy(Beta_t1)
        grad_Bt = -2 * x.T @ (y - x @ Beta_t)
        Beta_ht = Beta_t - mu * grad_Bt
        for j in range(p):
            # print(Beta_ht[j])
            Beta_t1[j] = max(1 - lamdda * mu / abs(Beta_ht[j]), 0) * Beta_ht[j]
        c = np.linalg.norm(Beta_t1 - Beta_t)
        if c < e:
            break
    rmse = sum((y - x @ Beta_t1) ** 2 / len(x)) ** 0.5
    return Beta_t1, a, rmse


# ADMM
def ADMM(x, y, lamdda, p, rho=0.01, e=1e-5):
    XtY = (X.T @ Y).reshape(-1, 1)
    R = np.linalg.inv(X.T @ X + rho * np.eye(p))
    Beta_t1 = np.zeros((p, 1))
    Z_t1 = np.zeros((p, 1))
    W_t1 = np.zeros((p, 1))
    for a in range(500000):  # 迭代上限次数
        Beta_t = copy.deepcopy(Beta_t1)
        Z_t = copy.deepcopy(Z_t1)
        W_t = copy.deepcopy(W_t1)
        U_t = rho * W_t
        Beta_t1 = R @ (XtY + rho * (Z_t - U_t))
        Z_t1 = S(Beta_t1 + W_t, lamdda / rho)
        W_t1 = W_t + Beta_t1 - Z_t1
        U_t1 = rho * W_t1
        if convergence(Beta_t1 - Beta_t, Z_t1 - Z_t, U_t1 - U_t, e):
            break
    rmse = sum((y - x @ Z_t1) ** 2 / len(x)) ** 0.5
    return Z_t1, a, rmse


p = X.shape[1]
Ip = np.eye(p)
# 选择最优λ
lam_err = {}
for m in range(10):
    train_X, train_Y, valid_X, valid_Y, n = k_fold_split(X_train, Y_train.reshape(-1, 1), m, 10)
    XtY = train_X.T @ train_Y
    for i in range(61):
        lamdda = 10 ** (-3 + 0.1 * i)
        R = np.linalg.inv(train_X.T @ train_X + lamdda * Ip)
        Beta = R @ XtY
        if i not in lam_err:
            lam_err[i] = (np.linalg.norm(valid_Y - valid_X @ Beta) ** 2) / n / 10
        else:
            lam_err[i] = (np.linalg.norm(valid_Y - valid_X @ Beta) ** 2) / n / 10
lam_err = sorted(lam_err.items(), key=lambda d:d[1])
opti_lam = 10 ** (-3 + 0.1 * lam_err[0][0])
print("optimal lambda:", opti_lam)

Beta_ADMM, c_ADMM, RMSE_ADMM = ADMM(X_train, Y_train.reshape(-1, 1), opti_lam, p, rho=1e-6, e=1e-10)
print('lasso_ADMM:\n', Beta_ADMM, '\n', 'iteration times:', c_ADMM, '\n', 'Training RMSE:', RMSE_ADMM)  # beta, 迭代次数，RMSE
MSE = 0
for i in range(len(X_test)):
    yiHat = np.sum(Beta_ADMM.reshape(-1) * X_test[i])
    MSE += ((yiHat - Y_test[i]) ** 2) / len(X_test)
print('Testing RMSE:', MSE ** 0.5)
plt.bar(range(len(Beta_ADMM.reshape(-1))), Beta_ADMM.reshape(-1))
plt.title('ADMM Lasso')
plt.show()

Beta_prox, c_prox, RMSE_prox = prox(X, Y.reshape(-1, 1), opti_lam, p, 1e-9, 1e-5)
print('lasso_proximal:\n', Beta_prox, '\n', c_prox, '\n', RMSE_prox)  # beta, 迭代次数，RMSE
MSE = 0
for i in range(len(X_test)):
    yiHat = np.sum(Beta_prox.reshape(-1) * X_test[i])
    MSE += ((yiHat - Y_test[i]) ** 2) / len(X_test)
print('Testing RMSE:', MSE ** 0.5)
plt.bar(range(len(Beta_prox.reshape(-1))), Beta_prox.reshape(-1), color='skyblue')
plt.title('proximal Lasso')
plt.show()

# 调包实验结果
from sklearn import linear_model

model = linear_model.LassoCV(random_state=0)
model.fit(X_train, Y_train)
print('skl.lasso:', model.alpha_)
print(model.coef_)
print("LassoCV Training RMSE:", mean_squared_error(Y_train, model.predict(X_train)) ** 0.5)
print("Testing RMSE:", mean_squared_error(Y_test, model.predict(X_test)) ** 0.5)
plt.bar(range(len(model.coef_)), model.coef_)
plt.title('skl.LassoCV')
plt.show()


## 回归树
def split_(X, i, thr):
    data_split = defaultdict(list)
    for x in X:
        data_split[x[i] < thr].append(x)
    return list(data_split.values()), list(data_split.keys())


def var_split(data, i, thr):
    left = []
    right = []
    for x in data:
        if x[i] >= thr:
            right.append(x)
        else:
            left.append(x)
    left = np.array(left)
    right = np.array(right)
    if len(left) < min_sample or len(right) < min_sample:
        return None
    return np.var(left[:, -1]) * left.shape[0] + np.var(right[:, -1]) * right.shape[0]


def feature_split(data):
    var = np.var(data[:, -1]) * data.shape[0]
    opti_v = float('inf')
    p = -1
    thr = None
    for i in range(len(data[0]) - 1):
        thrs = set(list(data[:, i]))
        for t in thrs:
            v = var_split(data, i, t)
            if v is None:
                continue
            if v < opti_v:
                opti_v, p, thr = v, i, t
    if var - opti_v < min_perform:
        return None, None
    return p, thr


def decision_tree(data):
    data = np.array(data)
    if data.shape[0] < min_sample:
        return np.mean(data[:, -1])
    p, thr = feature_split(data)
    if p is None:
        return thr
    nodes = {}
    nodes['feature'] = p
    nodes['threshold'] = thr
    split_data, values = split_(data, p, thr)
    for x, v in zip(split_data, values):
        nodes[v] = decision_tree(x)
    return nodes


def recur(tree, data):
    p = tree['feature']
    thr = tree['threshold']
    if isinstance(tree[data[p] < thr], dict):
        pred = recur(tree[data[p] < thr], data)
    else:
        pred = tree[data[p] < thr]
    if pred is None:
        for p in tree:
            if not isinstance(tree[p], dict):
                pred = tree[p]
                break
    return pred


def predict(tree, data):
    pred = []
    for x in data:
        y = recur(tree, x)
        pred.append(y)
    return np.array(pred)


min_perform = 2.5
min_sample = 10
Tree = decision_tree(X_train)
prediction1 = predict(Tree, X_train)
print("RegressionTree training", mean_squared_error(Y_train, prediction1) ** 0.5)
prediction2 = predict(Tree, X_test)
print("testing", mean_squared_error(Y_test, prediction2) ** 0.5)

# 调包
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)
y1_dtr = dtr.fit(X_train, Y_train).predict(X_train)
y2_dtr = dtr.predict(X_test)
print("DecisionTree training", mean_squared_error(Y_train, y1_dtr) ** 0.5)
print("testing", mean_squared_error(Y_test, y2_dtr) ** 0.5)
print(dtr.feature_importances_)
plt.bar(range(len(dtr.feature_importances_)), dtr.feature_importances_)
plt.title('DecisionTreeRegressor')
plt.show()

## 随机森林回归
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
y1_rfr = rfr.fit(X_train, Y_train).predict(X_train)
y2_rfr = rfr.predict(X_test)
print("RandomForest training", mean_squared_error(Y_train, y1_rfr) ** 0.5)
print("testing", mean_squared_error(Y_test, y2_rfr) ** 0.5)
print(rfr.feature_importances_)
plt.bar(range(len(rfr.feature_importances_)), rfr.feature_importances_)
plt.title('RandomForestRegressor')
plt.show()

# 极端森林回归
from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor(random_state=0)
y1_etr = etr.fit(X_train, Y_train).predict(X_train)
y2_etr = etr.predict(X_test)
print("ExtraTree training", mean_squared_error(Y_train, y1_etr) ** 0.5)
print("testing", mean_squared_error(Y_test, y2_etr) ** 0.5)
print(etr.feature_importances_)
plt.bar(range(len(etr.feature_importances_)), etr.feature_importances_)
plt.title('ExtraTreesRegressor')
plt.show()

# Adaboost回归
from sklearn.ensemble import AdaBoostRegressor
adb = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=0)
y1_adb = adb.fit(X_train, Y_train).predict(X_train)
y2_adb = adb.predict(X_test)
print("Adaboost training", mean_squared_error(Y_train, y1_adb) ** 0.5)
print("testing", mean_squared_error(Y_test, y2_adb) ** 0.5)
print(adb.feature_importances_)
plt.bar(range(len(adb.feature_importances_)), adb.feature_importances_)
plt.title('AdaBoostRegressor')
plt.show()

# XGboost回归
import xgboost as xgb
from xgboost import XGBRegressor
# from sklearn.model_selection import GridSearchCV

data = xgb.DMatrix(data=X_train, label=Y_train)
cv_results = xgb.cv(dtrain=data, params={'objective':'reg:linear', 'max_depth':10, 'alpha':10, 'learning_rate':0.1},
                    num_boost_round=50, early_stopping_rounds=10, metrics='rmse', seed=0, as_pandas=True)
cv_results.head()
print((cv_results['test-rmse-mean']).tail(1))
# param_ = {'n_estimators':range(80, 200, 4), 'max_depth':range(2, 15, 1)}
# grid = GridSearchCV(xgb, param_, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
# grid.fit(X_train, Y_train)
xg_reg = XGBRegressor(objective='reg:linear', max_depth=10).fit(X_train, Y_train)
y1_xgb = xg_reg.predict(X_train)
y2_xgb = xg_reg.predict(X_test)
print("XGB training", mean_squared_error(Y_train, y1_xgb) ** 0.5)
print("testing", mean_squared_error(Y_test, y2_xgb) ** 0.5)
print(xg_reg.feature_importances_)
plt.bar(range(len(xg_reg.feature_importances_)), xg_reg.feature_importances_)
plt.title('XGBRegressor')
plt.show()