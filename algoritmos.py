import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree, preprocessing
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

def regLineal():
    houses = pd.read_csv('./houses.csv')
    houses['ones']  = 1
    X = houses[['ones', '1stFlrSF', '2ndFlrSF', 'OverallQual', 'GarageArea']].values
    y = houses['SalePrice'].values

    model = LinearRegression()
    model.fit(X, y)

    coeficientes = list(model.coef_)
    coeficientes[0] = model.intercept_

    return coeficientes

def regLogistica(data):
    df = pd.read_csv('./diabetes.csv')

    X = df.iloc[:,[0,1,2,4,5,6,7]].values
    y = df.iloc[:, 8].values

    model = LogisticRegression(max_iter=len(X))
    model.fit(X, y)

    print(model.predict(data)[0])

def treeDecision(data):
    df = pd.read_csv('./titanic_train.csv')
    df['Sex'] = pd.get_dummies(df['Sex'])

    X = df[['Sex','Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    X = X.fillna(X.median()).values
    y = df['Survived'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y)

    model = tree.DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    

    resultado = model.predict(data)

    return resultado[0] 

def kmeans(data):
    df = datasets.load_iris()
    X_iris = df.data
    Y_iris = df.target
    
    X = pd.DataFrame(X_iris, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y = pd.DataFrame(Y_iris, columns=['Target'])

    model = KMeans(n_clusters=3, max_iter=1000)
    model.fit(X, y)

    y_kmeans = model.predict(X)

    accuracy = metrics.adjusted_rand_score(Y_iris, y_kmeans)

    resultado = model.predict(data)

    return resultado
