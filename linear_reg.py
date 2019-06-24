#Author: Peddi Karthik
"""
The libraries required for this code to run are:
-> numpy
-> matplotlib
-> pandas
-> sklearn
-> seaborn

To run this code a csv file path is needed as an input for the application of the
linear regression algorithm.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import os
from flask import Flask,render_template,request, redirect, url_for

def linear_reg1(dataset,target,dataset_file_name):
    dataset=dataset.dropna()
    cat_var=[]
    for i in dataset.columns.values:
        if len(set(dataset[i]))<=dataset.shape[0]//4:
            cat_var.append(i)
    cat_plots=[]
    for i in cat_var:
        if len(set(dataset[i]))<=10:
                ax=sns.countplot(x=i,data=dataset)
                plt.title('Values of Attribute \"'+i+"\"",loc='center')
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                filename='static/Plots/Linear Regression/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
                cat_plots.append(filename.replace(" ","%20"))
                plt.savefig(filename,bbox_inches='tight')
                plt.cla()

    X = dataset.iloc[:, dataset.columns != target]
    y = dataset[target]

    cols=list(X.columns)
    cols=[i for i in cols + list(cat_var) if i not in cols or i not in cat_var]
    trace=[]
    num_plots=[]
    for i in cols:
        trace0=go.Box(y=dataset[i].values,name=i)
        trace.append(trace0)
    layout = go.Layout(
        title=go.layout.Title(
            text='Boxplot of numerical variables',
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Attribute',
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Values',
            )
        ))
    fig = go.Figure(data=trace, layout=layout)
    filename='static/Plots/Linear Regression/'+dataset_file_name+'/Numerical Variables/Boxplot of numerical variables.html'
    num_plots.append(filename.replace(" ","%20"))
    fig=py.plot(fig, filename=filename,auto_open=False)


    for i in cat_var:
        X=pd.concat([X,pd.get_dummies(X[i],prefix=i)],axis=1,sort=False)
        X=X.drop([i],axis=1)

    cols=[1]
    cols+=list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)


    import statsmodels.formula.api as sm
    X = np.append ( arr = np.ones([X.shape[0],1]).astype(int), values = X, axis = 1)
    X_opt = X[:,:]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    flag=0
    to_keep=[i for i in range(X.shape[1])]
    dont_remove=[]
    while flag!=1:
        max_p=0
        ind=0
        for j in range(len(to_keep)):
            if regressor_OLS.pvalues[j]>0.05:
                if regressor_OLS.pvalues[j]>max_p and to_keep[j] not in dont_remove:
                    max_p=regressor_OLS.pvalues[j]
                    ind=j
        if max_p>0.05:
            inp=1
            if inp!=1:
                to_keep.remove(to_keep[ind])
            else:
                dont_remove.append(to_keep[ind])
                continue
        else:
            flag=0
            break
        if X_opt.shape[1]==len(to_keep):
            flag=0
        else:
            X_opt=X[:,to_keep]
            regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

    X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 1/3, random_state = 0)
    regressor_opt = LinearRegression()
    regressor_opt.fit(X_opt_train, y_opt_train)
     
    y_opt_pred = regressor_opt.predict(X_opt_test)

    var_score=metrics.explained_variance_score(y_opt_test, y_opt_pred)
    r2_score=metrics.r2_score(y_opt_test, y_opt_pred)
    return r2_score,var_score,num_plots,cat_plots
                
                    

