from flask import Flask, render_template,request,redirect,url_for
import prof_r as pr
import linear_reg as l
import logistic_reg as ll
import decision_tree as dt
import naive_bayes as nb
import random_forest as rf
import random_forest2 as rf2
import knn as kn
import kmeans as km
import os
import pandas as pd
import pandas_profiling
import numpy as np

app=Flask(__name__)
UPLOAD_FOLDER = '/datasets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['csv']

@app.route('/')
def main_page():
    return render_template('main_page.html')

@app.route('/linear_reg')
def linear_reg():
    return render_template('linear_regression.html')

@app.route('/linear_reg',methods = ['POST', 'GET'])
def linear1():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>Linear Regression</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/linear_reg");</script></body></html>'
        if not os.path.exists('static/Plots/Linear Regression/'+dataset_file_name):
            os.mkdir('static/Plots/Linear Regression/'+dataset_file_name)
            os.mkdir('static/Plots/Linear Regression/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/Linear Regression/'+dataset_file_name+"/Numerical Variables")
        profile_report_path="static/Plots/Linear Regression/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,0)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('linear_regression.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        dataset=pd.read_csv(dataset)
        dataset_file_name=request.form["dataset_file_name"]
        r2,var,num_plots,cat_plots=l.linear_reg1(dataset,target,dataset_file_name)
        return render_template('linear_regression.html',target=target,seq=2,var=var,r2=r2,num_plots=num_plots,cat_plots=cat_plots)

@app.route('/logistic_reg')
def logistic_reg():
    return render_template('logistic_regression.html')

@app.route('/logistic_reg',methods = ['POST', 'GET'])
def logistic1():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>Logistic Regression</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/logistic_reg");</script></body></html>'
        if not os.path.exists('static/Plots/Logistic Regression/'+dataset_file_name):
            os.mkdir('static/Plots/Logistic Regression/'+dataset_file_name)
            os.mkdir('static/Plots/Logistic Regression/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/Logistic Regression/'+dataset_file_name+"/Numerical Variables")
            os.mkdir('static/Plots/Logistic Regression/'+dataset_file_name+"/Metrics")
        profile_report_path="static/Plots/Logistic Regression/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,1)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('logistic_regression.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        no_of_labels=int(request.form["no_of_labels"])
        dataset=pd.read_csv(dataset)
        if target not in dataset.select_dtypes(include=['O']).columns.values:
            if len(set(dataset[target]))!=no_of_labels:
                return '<html><head><title>Logistic Regression</title></head><body><script>window.alert("The target variable must be categorical!");window.location.replace("/logistic_reg");</script></body></html>'
        dataset_file_name=request.form["dataset_file_name"]
        acc,f1,precision,recall,num_plots,cat_plots,conf_mat=ll.logistic_reg1(dataset,target,dataset_file_name)
        return render_template('logistic_regression.html',target=target,seq=2,acc=acc,precision=precision,recall=recall,f1=f1,num_plots=num_plots,cat_plots=cat_plots,conf_mat=conf_mat)

@app.route('/decision_tree')
def decision_tree():
    return render_template('decision_tree.html')

@app.route('/decision_tree',methods = ['POST', 'GET'])
def decision1():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>Decision Tree</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/decision_tree");</script></body></html>'
        if not os.path.exists('static/Plots/Decision Tree/'+dataset_file_name):
            os.mkdir('static/Plots/Decision Tree/'+dataset_file_name)
            os.mkdir('static/Plots/Decision Tree/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/Decision Tree/'+dataset_file_name+"/Numerical Variables")
            os.mkdir('static/Plots/Decision Tree/'+dataset_file_name+"/Metrics")
            os.mkdir('static/Plots/Decision Tree/'+dataset_file_name+"/Decision Tree")
        profile_report_path="static/Plots/Decision Tree/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,1)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('decision_tree.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        no_of_labels=int(request.form["no_of_labels"])
        dataset=pd.read_csv(dataset)
        if target not in dataset.select_dtypes(include=['O']).columns.values:
            if len(set(dataset[target]))!=no_of_labels:
                return '<html><head><title>Decision Regression</title></head><body><script>window.alert("The target variable must be categorical!");window.location.replace("/decision_tree");</script></body></html>'
        dataset_file_name=request.form["dataset_file_name"]
        acc,f1,precision,recall,num_plots,cat_plots,conf_mat,dtree=dt.decision_tree1(dataset,target,dataset_file_name)
        return render_template('decision_tree.html',target=target,seq=2,acc=acc,precision=precision,recall=recall,f1=f1,num_plots=num_plots,cat_plots=cat_plots,conf_mat=conf_mat,dtree=dtree)


@app.route('/naive_bayes')
def naive_bayes():
    return render_template('naive_bayes.html')

@app.route('/naive_bayes',methods = ['POST', 'GET'])
def naive1():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>Naive Bayes</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/naive_bayes");</script></body></html>'
        if not os.path.exists('static/Plots/Naive Bayes/'+dataset_file_name):
            os.mkdir('static/Plots/Naive Bayes/'+dataset_file_name)
            os.mkdir('static/Plots/Naive Bayes/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/Naive Bayes/'+dataset_file_name+"/Numerical Variables")
            os.mkdir('static/Plots/Naive Bayes/'+dataset_file_name+"/Metrics")
        profile_report_path="static/Plots/Naive Bayes/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,1)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('naive_bayes.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        no_of_labels=int(request.form["no_of_labels"])
        dataset=pd.read_csv(dataset)
        if target not in dataset.select_dtypes(include=['O']).columns.values:
            if len(set(dataset[target]))!=no_of_labels:
                return '<html><head><title>Naive Bayes</title></head><body><script>window.alert("The target variable must be categorical!");window.location.replace("/naive_bayes");</script></body></html>'
        dataset_file_name=request.form["dataset_file_name"]
        acc,f1,precision,recall,num_plots,cat_plots,conf_mat=nb.naive_bayes1(dataset,target,dataset_file_name)
        return render_template('naive_bayes.html',target=target,seq=2,acc=acc,precision=precision,recall=recall,f1=f1,num_plots=num_plots,cat_plots=cat_plots,conf_mat=conf_mat)

@app.route('/random_forest')
def random_forest():
    return render_template('random_forest.html')

@app.route('/random_forest',methods = ['POST', 'GET'])
def random1():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>Random Forest</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/random_forest");</script></body></html>'
        if not os.path.exists('static/Plots/Random Forest/'+dataset_file_name):
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name)
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name+"/Numerical Variables")
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name+"/Metrics")
        profile_report_path="static/Plots/Random Forest/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,1)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('random_forest.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        no_of_labels=int(request.form["no_of_labels"])
        dataset=pd.read_csv(dataset)
        if target not in dataset.select_dtypes(include=['O']).columns.values:
            if len(set(dataset[target]))!=no_of_labels:
                return '<html><head><title>Random Forest</title></head><body><script>window.alert("The target variable must be categorical!");window.location.replace("/random_forest");</script></body></html>'
        dataset_file_name=request.form["dataset_file_name"]
        acc,f1,precision,recall,num_plots,cat_plots,conf_mat,mae=rf.random_forest1(dataset,target,dataset_file_name)
        return render_template('random_forest.html',target=target,seq=2,acc=acc,precision=precision,recall=recall,f1=f1,num_plots=num_plots,cat_plots=cat_plots,conf_mat=conf_mat,mae=mae)

@app.route('/Knn')
def Knn():
    return render_template('Knn.html')

@app.route('/Knn',methods = ['POST', 'GET'])
def Knn1():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>k-Nearest Neighbours</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/Knn");</script></body></html>'
        if not os.path.exists('static/Plots/Knn/'+dataset_file_name):
            os.mkdir('static/Plots/Knn/'+dataset_file_name)
            os.mkdir('static/Plots/Knn/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/Knn/'+dataset_file_name+"/Numerical Variables")
            os.mkdir('static/Plots/Knn/'+dataset_file_name+"/Metrics")
        profile_report_path="static/Plots/Knn/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,1)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('Knn.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        no_of_labels=int(request.form["no_of_labels"])
        dataset=pd.read_csv(dataset)
        if target not in dataset.select_dtypes(include=['O']).columns.values:
            if len(set(dataset[target]))!=no_of_labels:
                return '<html><head><title>k-Nearest Neighbours</title></head><body><script>window.alert("The target variable must be categorical!");window.location.replace("/Knn");</script></body></html>'
        dataset_file_name=request.form["dataset_file_name"]
        acc,f1,precision,recall,num_plots,cat_plots,conf_mat,er,ac_vs_k=kn.knn1(dataset,target,dataset_file_name)
        return render_template('Knn.html',target=target,seq=2,acc=acc,precision=precision,recall=recall,f1=f1,num_plots=num_plots,cat_plots=cat_plots,conf_mat=conf_mat,er=er,ac_vs_k=ac_vs_k)

@app.route('/kmeans')
def Kmeans():
    return render_template('kmeans.html')

@app.route('/kmeans',methods = ['POST', 'GET'])
def k_means1():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>K-Means</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/kmeans");</script></body></html>'
        if not os.path.exists('static/Plots/K-means/'+dataset_file_name):
            os.mkdir('static/Plots/K-means/'+dataset_file_name)
            os.mkdir('static/Plots/K-means/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/K-means/'+dataset_file_name+"/Numerical Variables")
            os.mkdir('static/Plots/K-means/'+dataset_file_name+"/Metrics")
        profile_report_path="static/Plots/K-means/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,1)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('kmeans.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        no_of_labels=int(request.form["no_of_labels"])
        dataset=pd.read_csv(dataset)
        if target not in dataset.select_dtypes(include=['O']).columns.values:
            if len(set(dataset[target]))!=no_of_labels:
                return '<html><head><title>K-Means</title></head><body><script>window.alert("The target variable must be categorical!");window.location.replace("/kmeans");</script></body></html>'
        dataset_file_name=request.form["dataset_file_name"]
        nmi,ars,ami,cat_plots,conf_mat,scat_plots,wc_plot=km.k_means1(dataset,target,dataset_file_name)
        return render_template('kmeans.html',target=target,seq=2,nmi=nmi,ars=ars,ami=ami,scat_plots=scat_plots,cat_plots=cat_plots,conf_mat=conf_mat,wc_plot=wc_plot)

@app.route('/random_forest2')
def random_forest2():
    return render_template('random_forest2.html')

@app.route('/random_forest2',methods = ['POST', 'GET'])
def random2():
    if "target" not in request.form:
        file=request.files['dataset']
        if allowed_file(file.filename):
            file.save(os.path.join("static/datasets/", file.filename))
            dataset_file_path="static/datasets/"+file.filename
            dataset=pd.read_csv(dataset_file_path)
            dataset_file_name=file.filename[:-4]
        else:
            return '<html><head><title>Random Forest Regression</title></head><body><script>window.alert("The file uploaded should be csv file");window.location.replace("/random_forest2");</script></body></html>'
        if not os.path.exists('static/Plots/Random Forest/'+dataset_file_name):
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name)
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name+"/Categorical Variables")
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name+"/Numerical Variables")
            os.mkdir('static/Plots/Random Forest/'+dataset_file_name+"/Metrics")
        profile_report_path="static/Plots/Random Forest/"+dataset_file_name+"/profile_report.html"
        cols=pr.prof(dataset,profile_report_path,0)        
        profile_report_path=profile_report_path.replace(" ","%20")
        return render_template('random_forest2.html',dataset=dataset_file_path,cols=cols,profile_report_path=profile_report_path,dataset_file_name=dataset_file_name)
    else:
        target=request.form["target"]
        dataset=request.form["dataset"]
        dataset=pd.read_csv(dataset)
        dataset_file_name=request.form["dataset_file_name"]
        mae,var_score,r2_score,num_plots,cat_plots,conf_mat=rf2.random_forest2(dataset,target,dataset_file_name)
        return render_template('random_forest2.html',target=target,seq=2,mae=mae,var_score=var_score,r2_score=r2_score,num_plots=num_plots,cat_plots=cat_plots,conf_mat=conf_mat)


@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')

@app.route('/learn_linear')
def learn_linear():
    return render_template('learn_x.html',filename="static/Linear Regression.PNG",name="Linear Regression")

@app.route('/learn_logistic')
def learn_logistic():
    return render_template('learn_x.html',filename="static/Logistic Regression.PNG",name="Logistic Regression")

@app.route('/learn_decision')
def learn_decision():
    return render_template('learn_x.html',filename="static/Decision Tree.PNG",name="Decision Tree")

@app.route('/learn_random')
def learn_random():
    return render_template('learn_x.html',filename="static/Random Forest.PNG",name="Random Forest")

@app.route('/learn_kmeans')
def learn_kmeans():
    return render_template('learn_x.html',filename="static/K Means.PNG",name="K Means")

@app.route('/learn_knn')
def learn_knn():
    return render_template('learn_x.html',filename="static/Knn.PNG",name="k-NN")

if __name__ == '__main__':
    app.run()
