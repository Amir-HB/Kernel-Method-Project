import subprocess
# subprocess.call('reset')

import numpy as np
from numpy import *
import csv
import pandas as pd
import subprocess

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from cvxopt import matrix, solvers
from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
from sklearn.decomposition import PCA
from itertools import product
from numpy import linalg as LA

# import pylab

X0 = pd.read_csv('Xtr0.csv', sep='\s+', header=None)
X0 = np.squeeze(np.array(X0))
X1 = pd.read_csv('Xtr1.csv', sep='\s+', header=None)
X1 = np.squeeze(np.array(X1))
X2 = pd.read_csv('Xtr2.csv', sep='\s+', header=None)
X2 = np.squeeze(np.array(X2))
X = np.concatenate((X0, X1, X2))
n = X.shape[0]

X_test0 = pd.read_csv('Xte0.csv', sep='\s+', header=None)
X_test0 = np.squeeze(np.array(X_test0))
X_test1 = pd.read_csv('Xte1.csv', sep='\s+', header=None)
X_test1 = np.squeeze(np.array(X_test1))
X_test2 = pd.read_csv('Xte2.csv', sep='\s+', header=None)
X_test2 = np.squeeze(np.array(X_test2))
X_test = np.concatenate((X_test0, X_test1, X_test2))
n_test = X_test.shape[0]

y0 = pd.read_csv('Ytr0.csv', usecols=[1])
y0 = np.squeeze(np.array(y0))
y0 = 2*y0-1
y1 = pd.read_csv('Ytr1.csv', usecols=[1])
y1 = np.squeeze(np.array(y1))
y1 = 2*y1-1
y2 = pd.read_csv('Ytr2.csv', usecols=[1])
y2 = np.squeeze(np.array(y2))
y2 = 2*y2-1
y = np.concatenate((y0, y1, y2))

Xnew = np.asmatrix(X)


### Spectrum Kernel function
def spectrum_phi(x, l):
    n = len(x)
    dic = {}
    for i in range(0,n-l):
    	s = x[i:i+l]
    	if s in dic.keys(): dic.update({s:dic[s]+1})
    	else: dic.update({s:1})
    return dic

def spectrum_prod(x_d, y_d):
    s = 0
    for k in x_d.keys():
        if k in y_d.keys(): s = s + x_d[k] * y_d[k]
    return s

def kernel_spectrum(X, l):
    n = X.shape[0]
    phi = np.vectorize(lambda x: spectrum_phi(x, l))
    phi_X = phi(X)
    K = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i + 1):
            K[i, j] = spectrum_prod(phi_X[i], phi_X[j])
            K[j, i] = K[i, j]
    return K

#K = kernel_spectrum(X,4)
#df = pd.DataFrame(K)
#df.to_csv("kernel_spectrum_X_l4.csv",index=False, sep=";",line_terminator="\r\n")
phi = np.vectorize(lambda x: spectrum_phi(x, 6))

XX = np.concatenate((X, X_test))
pXX = phi(XX)
pXX = pXX.tolist()
df = pd.DataFrame(pXX)
pXX = df.values
where_are_NaNs = isnan(pXX)
pXX[where_are_NaNs] = 0
pX1 = pXX[0:2000]/91.2
pX2 = pXX[2000:4000]/90
pX3 = pXX[4000:6000]/93.1



pX_test1 = pXX[n:n+1000]/91.2
pX_test2 = pXX[n+1000:n+2000]/90
pX_test3 = pXX[n+2000:n+3000]/93.1



X_train1 = pX1
X_train2 = pX2
X_train3 = pX3

y_train1 = y0
y_train2 = y1
y_train3 = y2

X_test1 = pX_test1
X_test2 = pX_test2
X_test3 = pX_test3



    




#K = np.squeeze(np.array(K))
K1 = np.dot(X_train1,X_train1.transpose())
m=X_train1.shape[0]
U = np.asmatrix((1 / m) * (np.ones((m, m))))
KC1 = np.dot(np.dot(np.identity(m) - U, K1), np.identity(m) - U)

###

P1 = matrix(KC1, tc='d')
G1 = matrix(np.concatenate((np.diag(y_train1), np.diag(-y_train1))), tc='d')
q1 = matrix( -y_train1, tc='d')
lambd = 100
C=4.1/100
h1 = matrix(np.concatenate((np.ones((X_train1.shape[0], 1)) / C, np.zeros((X_train1.shape[0], 1)))))

alpha = solvers.qp(P1, q1, G1, h1)
a1 = alpha['x']
#print("ass5")
X_testC1 = X_test1 - X_train1.mean(axis=0)
X_C1 = X_train1 - X_train1.mean(axis=0)

    #lin_clf = svm.LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
           #   intercept_scaling=1, loss='squared_hinge', max_iter=1000,
           #   multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
            #  verbose=0)

    #lin_clf.fit(X_C , y_train)
    #pred=lin_clf.predict(X_testC)

J1=np.dot(X_C1,X_testC1.transpose())



f1 = np.dot(np.asmatrix(a1).transpose(), J1)
pred1 = np.sign(f1)
pred1=(pred1 + 1) / 2


K2 = np.dot(X_train2, X_train2.transpose())
m = X_train2.shape[0]
U = np.asmatrix((1 / m) * (np.ones((m, m))))
KC2 = np.dot(np.dot(np.identity(m) - U, K2), np.identity(m) - U)

    ###

P2 = matrix(KC2, tc='d')
G2 = matrix(np.concatenate((np.diag(y_train2), np.diag(-y_train2))), tc='d')
q2 = matrix(-y_train2, tc='d')
lambd = 100
C = 2.96 / 100
h2 = matrix(np.concatenate((np.ones((X_train2.shape[0], 1)) / C, np.zeros((X_train2.shape[0], 1)))))

alpha = solvers.qp(P2, q2, G2, h2)
a2 = alpha['x']
    # print("ass5")
X_testC2 = X_test2 - X_train2.mean(axis=0)
X_C2 = X_train2 - X_train2.mean(axis=0)

    # lin_clf = svm.LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
    #   intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    #   multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #  verbose=0)

    # lin_clf.fit(X_C , y_train)
    # pred=lin_clf.predict(X_testC)

J2 = np.dot(X_C2, X_testC2.transpose())

f2 = np.dot(np.asmatrix(a2).transpose(), J2)
pred2 = np.sign(f2)
pred2=(pred2+1)/2


K3 = np.dot(X_train3, X_train3.transpose())
m = X_train3.shape[0]
U = np.asmatrix((1 / m) * (np.ones((m, m))))
KC3 = np.dot(np.dot(np.identity(m) - U, K3), np.identity(m) - U)

P3 = matrix(KC3, tc='d')
G3 = matrix(np.concatenate((np.diag(y_train3), np.diag(-y_train3))), tc='d')
q3 = matrix(-y_train3, tc='d')
lambd = 100
C=4.2  / 100
h3 = matrix(np.concatenate((np.ones((X_train3.shape[0], 1)) / C, np.zeros((X_train3.shape[0], 1)))))

alpha = solvers.qp(P3, q3, G3, h3)
a3 = alpha['x']
    # print("ass5")
X_testC3 = X_test3 - X_train3.mean(axis=0)
X_C3 = X_train3 - X_train3.mean(axis=0)

    # lin_clf = svm.LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
    #   intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    #   multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #  verbose=0)

    # lin_clf.fit(X_C , y_train)
    # pred=lin_clf.predict(X_testC)

J3 = np.dot(X_C3, X_testC3.transpose())

f3 = np.dot(np.asmatrix(a3).transpose(), J3)
pred3 = np.sign(f3)
pred3=(pred3+1)/2  ###





pred=np.concatenate((pred1, pred2, pred3),axis=1)



output = np.zeros((n_test, 2))
for i in range(0, n_test):
    output[i, 0] = int(i)
    output[i, 1] = pred[0,i]

df = pd.DataFrame(output, columns=['Id', 'Bound'])
df = df.fillna(0)
df = df.astype(int)
df.to_csv("last1.csv", index=False, sep=",")







