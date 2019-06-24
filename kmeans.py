import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import pandas as pd
import numpy as np

import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px
import os

def k_means1(dataset,target,dataset_file_name):
    dataset=dataset.fillna(dataset.mean())
    cat_plots=[]
    cat_var=[]
    for i in dataset.columns.values:
        if len(set(dataset[i]))<=dataset.shape[0]//4:
            cat_var.append(i)
    for i in cat_var or i==target:
        if len(set(dataset[i]))<=10:
            ax=sns.countplot(x=i,data=dataset)
            plt.title('Values of Attribute \"'+i+"\"",loc='center')
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.tight_layout()
            filename='static/Plots/K-means/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
            cat_plots.append(filename.replace(" ","%20"))
            plt.savefig(filename,bbox_inches='tight',dpi=600)
            plt.cla()

    def plot_scatterplot(dataset,cat_var,target):
        scat_plots=[]
        columns=dataset.columns
        for i in range(len(columns)):
            for j in range(i+1,len(columns)):
                if columns[i] not in cat_var and columns[j] not in cat_var and columns[i]!=target and columns[j]!=target:
                    sns.scatterplot(x=columns[i],y=columns[j],hue=target,data=dataset)
                    plt.title("Scatter plot of \""+columns[i]+"\" VS \""+columns[j]+"\"")
                    plt.xlabel(columns[i])
                    plt.ylabel(columns[j])
                    filename="static/Plots/K-means/"+dataset_file_name+"/Numerical Variables/Scatterplot of Attribute"+columns[i]+" and "+columns[j]+".png"
                    scat_plots.append(filename.replace(" ","%20"))
                    plt.savefig(filename,bbox_inches='tight')
                    plt.cla()
                    plt.close()
        return scat_plots

    scat_plots=plot_scatterplot(dataset,cat_var,target)

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    cat_var=dataset.select_dtypes(include=['O']).columns.values
    for i in cat_var:
        labelencoder.fit(list(dataset[i]))
        dataset[i] = labelencoder.transform(list(dataset[i]))
        
    X = dataset.iloc[:, dataset.columns != target]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training and 30% test

    wcss=[]
    for i in range(1,11):
        model = KMeans(n_clusters=i)
        model.fit(X_train)
        wcss.append(model.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    filename="static/Plots/K-means/"+dataset_file_name+"/Metrics/Optimal Number of cluster based on WCSS.png"
    wc_plot=filename.replace(" ","%20")
    plt.savefig("static/Plots/K-means/"+dataset_file_name+"/Metrics/Optimal Number of cluster based on WCSS.png")
    plt.cla()
    plt.close()
    max=0
    op_k=0
    for i in range(2,40):
        model = KMeans(n_clusters=i)
        model.fit(X_train)
        y_pred=model.predict(X_test)
        nmi=metrics.normalized_mutual_info_score(y_test, y_pred)
        if nmi>max:
            max=nmi
            op_k=i
    model = KMeans(n_clusters=op_k)
    model.fit(X_train)
    y_pred=model.predict(X_test)
    nmi=metrics.normalized_mutual_info_score(y_pred,y_test)
    ars=metrics.adjusted_rand_score(y_pred,y_test)
    ami=metrics.adjusted_mutual_info_score(y_pred,y_test)
    conf_mat=[]
    if len(set(dataset[target]))<=10:
        ax=sns.heatmap(metrics.confusion_matrix(y_test,y_pred),annot=True)
        ax.set_title('Confusion Matrix of Trained K means Classifier',loc='center')
        ax.set_xlabel('Actual target value')
        ax.set_ylabel('Predicted Value')
        filename='static/Plots/K-means/'+dataset_file_name+'/Metrics/confusion_matrix.png'
        conf_mat.append(filename.replace(" ","%20"))
        plt.savefig('static/Plots/K-means/'+dataset_file_name+'/Metrics/confusion_matrix.png',dpi=1000)
        plt.cla()
    return nmi,ars,ami,cat_plots,conf_mat,scat_plots,wc_plot


