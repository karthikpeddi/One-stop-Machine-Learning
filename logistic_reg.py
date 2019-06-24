import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.graph_objs import *
from flask import Flask,render_template,request, redirect, url_for

def logistic_reg1(dataset,target,dataset_file_name):
    dataset=dataset.dropna()
    def plot_boxplot(dataset,cat_var):
        cols=list(dataset.columns)
        cols=[i for i in cols + list(cat_var) if i not in cols or i not in cat_var]
        trace=[]
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
        fig=py.plot(fig, filename='static/Plots/Logistic Regression/'+dataset_file_name+'/Numerical Variables/Boxplot of numerical variables.html',auto_open=False)
        return
    
    num_plots=[]
    filename1='static/Plots/Logistic Regression/'+dataset_file_name+'/Numerical Variables/Boxplot of numerical variables.html'
    num_plots.append(filename1.replace(" ","%20"))
    cat_plots=[]
    cat_var=[]
    for i in dataset.columns.values:
        if len(set(dataset[i]))<=dataset.shape[0]//4:
            cat_var.append(i)
    for i in cat_var:
        if i!=target:
            if len(set(dataset[i]))<=10:
                ax=sns.countplot(x=i,data=dataset)
                plt.title('Values of Attribute \"'+i+"\"",loc='center')
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                filename='static/Plots/Logistic Regression/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
                cat_plots.append(filename.replace(" ","%20"))
                plt.savefig(filename,bbox_inches='tight')
                plt.cla()

    i=target
    if len(set(dataset[i]))<=10:
        ax=sns.countplot(x=i,data=dataset)
        plt.title('Values of Attribute \"'+i+"\"",loc='center')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()
        filenams='static/Plots/Logistic Regression/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
        cat_plots.append(filename.replace(" ","%20"))
        plt.savefig(filename,bbox_inches='tight',dpi=600)
        plt.cla()
        plt.close()
    
    plot_boxplot(dataset,cat_var)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    cat_var=dataset.select_dtypes(include=['O']).columns.values
    for i in cat_var:
        labelencoder.fit(list(dataset[i]))
        dataset[i] = labelencoder.transform(list(dataset[i]))
            
    X = dataset.iloc[:, dataset.columns != target]
    y = dataset[target]
    X = np.append ( arr = np.ones([X.shape[0],1]).astype(int), values = X, axis = 1)
    X_opt = X[:,:]
    import statsmodels.api as sm
    logit_model=sm.Logit(y,X)
    result=logit_model.fit()
    flag=0
    to_keep=[i for i in range(X.shape[1])]
    dont_remove=[]
    while flag!=1:
        max_p=0
        ind=0
        for j in range(len(to_keep)):
            if result.pvalues[j]>0.05:
                if result.pvalues[j]>max_p and to_keep[j] not in dont_remove:
                    max_p=result.pvalues[j]
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
            result = sm.Logit(endog = y, exog = X_opt).fit()

    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 1/3, random_state = 0)
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter=200).fit(X_train, y_train)
    y_pred=mul_lr.predict(X_test)
    conf_mat=[]
    if len(set(dataset[target]))<=10:
        ax=sns.heatmap(metrics.confusion_matrix(y_test,y_pred),annot=True)
        ax.set_title('Confusion Matrix of Trained Logistic Regression Classifier',loc='center')
        ax.set_xlabel('Actual target value')
        ax.set_ylabel('Predicted Value')
        filename='static/Plots/Logistic Regression/'+dataset_file_name+'/Metrics/confusion_matrix.png'
        conf_mat.append(filename.replace(" ","%20"))
        plt.savefig(filename,dpi=1000)
        plt.cla()
        plt.close()
    acc=metrics.accuracy_score(y_test, y_pred)
    if len(set(dataset[target]))!=2:
        f1=metrics.f1_score(y_test, y_pred,average="macro")
        precision=metrics.precision_score(y_test, y_pred,average="macro")
        recall=metrics.recall_score(y_test, y_pred,average="macro")
    else:
        f1=metrics.f1_score(y_test, y_pred)
        precision=metrics.precision_score(y_test, y_pred)
        recall=metrics.recall_score(y_test, y_pred)
    return acc,f1,precision,recall,num_plots,cat_plots,conf_mat
