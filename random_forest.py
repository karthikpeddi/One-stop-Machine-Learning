import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from feature_selector import FeatureSelector
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from plotly.graph_objs import *
import os

def random_forest1(dataset,target,dataset_file_name):
    dataset=dataset.fillna(dataset.mean())
    cat_var=dataset.select_dtypes(include=['O']).columns.values
    cat_plots=[]
    for i in cat_var:
        if i!=target:
            if len(set(dataset[i]))<=10:
                ax=sn.countplot(x=i,data=dataset)
                plt.title('Values of Attribute \"'+i+"\"",loc='center')
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                plt.tight_layout()
                filename='static/Plots/Random Forest/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
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
        filename='static/Plots/Random Forest/'+dataset_file_name+'/Categorical Variables/Attribute '+i+'.png'
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training and 30% test

    rf=RandomForestClassifier(n_estimators=1000,random_state=42)
    rf.fit(X_train,y_train)
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_cols, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    fi=[]
    for i in feature_importances:
        if i[1]>0.05:
            fi.append(i[0])
    rf_most_important = RandomForestClassifier(n_estimators= 1000, random_state=42)
    X_train = X_train[fi]
    X_test = X_test[fi]
    rf_most_important.fit(X_train, y_train)
    y_pred=rf_most_important.predict(X_test)
    errors = abs(y_pred - y_test)
    mae=round(np.mean(errors), 2)
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
        ax.set_title('Confusion Matrix of Trained Random Forest Classifier',loc='center')
        ax.set_xlabel('Actual target value')
        ax.set_ylabel('Predicted Value')
        filename='static/Plots/Random Forest/'+dataset_file_name+'/Metrics/confusion_matrix.png'
        conf_mat.append(filename.replace(" ","%20"))
        plt.savefig(filename,dpi=1000)
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
        py.plot(fig, filename='static/Plots/Random Forest/'+dataset_file_name+'/Numerical Variables/Boxplot of numerical variables.html',auto_open=False)
        return
    plot_boxplot(dataset,cat_var,target)
    num_plots=[]
    filename1='static/Plots/Random Forest/'+dataset_file_name+'/Numerical Variables/Boxplot of numerical variables.html'
    num_plots.append(filename1.replace(" ","%20"))
    return acc,f1,precision,recall,num_plots,cat_plots,conf_mat,mae
