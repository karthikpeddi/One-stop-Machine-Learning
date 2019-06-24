import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from feature_selector import FeatureSelector
from sklearn.preprocessing import StandardScaler  
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from plotly.graph_objs import *
import os

def knn1(dataset,target,dataset_file_name):
    dataset=dataset.fillna(dataset.mean())
    cat_var=[]
    for i in dataset.columns.values:
        if len(set(dataset[i]))<=dataset.shape[0]//4:
            cat_var.append(i)
    cat_plots=[]
    for i in cat_var:
        if i!=target:
            if len(set(dataset[i]))<=10:
                ax=sn.countplot(x=i,data=dataset)
                plt.title('Values of Attribute \"'+i+"\"",loc='center')
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                plt.tight_layout()
                filename='static/Plots/Knn/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
                cat_plots.append(filename.replace(" ","%20"))
                plt.savefig(filename,bbox_inches='tight',dpi=600)
                plt.cla()
                plt.close()
    i=target
    if len(set(dataset[i]))<=10:
        ax=sn.countplot(x=i,data=dataset)
        plt.title('Values of Attribute \"'+i+"\"",loc='center')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()
        filename='static/Plots/Knn/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
        cat_plots.append(filename.replace(" ","%20"))
        plt.savefig(filename,bbox_inches='tight',dpi=600)
        plt.cla()
        plt.close()
            
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    cat_var=dataset.select_dtypes(include=['O']).columns.values
    for i in cat_var:
        labelencoder.fit(list(dataset[i]))
        dataset[i] = labelencoder.transform(list(dataset[i]))

    feature_cols=[i for i in dataset.columns.values if i!=target]
    fs = FeatureSelector(data = dataset, labels = feature_cols)
    fs.identify_missing(missing_threshold = 0.3)
    missing_features = fs.ops['missing']
    feature_cols=[i for i in feature_cols if i not in missing_features]
    fs = FeatureSelector(data = dataset, labels = feature_cols)
    fs.identify_collinear(correlation_threshold = 0.98)
    collinear_features = fs.ops['collinear']
    feature_cols=[i for i in feature_cols if i not in collinear_features]
    fs = FeatureSelector(data = dataset, labels = feature_cols)
    fs.identify_single_unique()
    single_unique=fs.ops['single_unique']
    feature_cols=[i for i in feature_cols if i not in single_unique]
    X=dataset[feature_cols]
    y=dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
    scaler = StandardScaler()  
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  

    min=1
    optimal_k=0
    error_rate=[]
    accuracies=[]
    for i in range(1, 40):  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error_rate.append(np.mean(y_pred != y_test))
        accuracies.append(metrics.accuracy_score(y_pred,y_test))
        if np.mean(y_pred != y_test)<min:
            min=np.mean(y_pred != y_test)
            optimal_k=i

    er=""
    ac_vs_k=""
    fig=plt.figure()  
    plt.plot(range(1, 40), error_rate, color='red', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rates for different k')  
    plt.xlabel('K Value')  
    plt.ylabel('Mean Error Rate')
    er="static/Plots/Knn/"+dataset_file_name+"/Metrics/error_rate.png"
    plt.savefig("static/Plots/Knn/"+dataset_file_name+"/Metrics/error_rate.png",dpi=1000)
    plt.cla()
    plt.close()
    plt.figure()  
    plt.plot(range(1, 40), accuracies, color='blue',marker='o',  
             markerfacecolor='blue', markersize=10)
    plt.title('Accuracy of the model for different k')  
    plt.xlabel('K Value')  
    plt.ylabel('Accuracy')
    ac_vs_k="static/Plots/Knn/"+dataset_file_name+"/Metrics/accuracy_vs_k.png"
    plt.savefig("static/Plots/Knn/"+dataset_file_name+"/Metrics/accuracy_vs_k.png",dpi=1000)
    plt.cla()
    plt.close()

    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc=metrics.accuracy_score(y_test, y_pred)
    if len(set(dataset[target]))!=2:
        f1=metrics.f1_score(y_test, y_pred,average="macro")
        precision=metrics.precision_score(y_test, y_pred,average="macro")
        recall=metrics.recall_score(y_test, y_pred,average="macro")
    else:
        f1=metrics.f1_score(y_test, y_pred)
        precision=metrics.precision_score(y_test, y_pred)
        recall=metrics.recall_score(y_test, y_pred)

    conf_mat=[]
    if len(set(dataset[target]))<=10:
        ax=sn.heatmap(metrics.confusion_matrix(y_test,y_pred),annot=True)
        ax.set_title('Confusion Matrix of Trained k-NN Classifier',loc='center')
        ax.set_xlabel('Actual target value')
        ax.set_ylabel('Predicted Value')
        filename='static/Plots/Knn/'+dataset_file_name+'/Metrics/confusion_matrix.png'
        conf_mat.append(filename.replace(" ","%20"))
        plt.savefig('static/Plots/Knn/'+dataset_file_name+'/Metrics/confusion_matrix.png',dpi=1000)
        plt.cla()
        plt.close()

    def plot_boxplot(dataset,cat_var,target):
        cols=list(dataset.columns)
        cols=[i for i in cols if i not in cat_var and i!=target]
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
        py.plot(fig, filename='static/Plots/Knn/'+dataset_file_name+'/Numerical Variables/Boxplot of numerical variables.html',auto_open=False)
        return
    plot_boxplot(dataset,cat_var,target)
    num_plots=[]
    filename1='static/Plots/Knn/'+dataset_file_name+'/Numerical Variables/Boxplot of numerical variables.html'
    num_plots.append(filename1.replace(" ","%20"))
    return acc,f1,precision,recall,num_plots,cat_plots,conf_mat,er,ac_vs_k
