from flask import Flask, render_template,request,session,flash
import os,glob,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score
import cls as cl
import sl


app = Flask(__name__)
app.config['SECRET_KEY']='password'
@app.route('/feature_selector_tool', methods=["GET","POST"])
def feature_selector_tool():
    if request.method =="POST":
        
        if request.form["form_name"]=="form1":
            file = request.files["file"]
            file.save(os.path.join("uploads",file.filename))
            dataset = pd.read_csv(os.path.join("uploads",file.filename))
            data_summary = Summarize_data(dataset)
            path = os.path.join("uploads",file.filename)
            path_dict = {"path":path,"name":file.filename}
            outfile = open('datastore.json','w')
            json.dump(path_dict,outfile)
            outfile.close()
            return render_template('Feature Selection.html',message = " ",files_complete = data_summary,files = data_summary, metric = " ",metric_trim = " ", plot = " ",visualize=0)
            #return render_template('home.html',files = data_summary)
            ###rank = filter(request.form["feature"],dataset)
           
        elif request.form["form_name"]=="form2":
            classification_summ_trim = " "
            data_summary_trim = " "
            message=" "
            reorder = 0
            trim=0
            if request.form["Check"] == "enable":
                infile = open('datastore.json','r')
                path_dict = json.load(infile)
                infile.close()
                feature = request.form["feature_selectors"]
                n,data_new=sl.filter(feature,path_dict['path'])

                #acc=cl.model(path_dict['path'],session['classifier'])
                message = "You just uploaded the {} file!\n The important features of the given dataset in descending order are {}".format(path_dict['name'],n)
                #path = Reorder()
                path = os.path.join("uploads","reorderdata.csv")
                data_new.to_csv(path,index= False)
                path_dict.update({"reorder":path})
                outfile = open('datastore.json','w')
                json.dump(path_dict,outfile)
                outfile.close()
                reorder=1
            
            if request.form["select_data"] == "1":
                cols_required = int(request.form["ranking"])
                #trim dataset
                Trim_dataset(reorder,cols_required)
                trim=1
            if trim!=0 or reorder!=0:
                classification_summ_trim,data_summary_trim = Classify(request.form["classifier"],trim,reorder )
            classification_summary,data_summary = Classify(request.form["classifier"],0 ,0)
            if data_summary_trim == " ":
                data_summary_trim = data_summary 
            return render_template('Feature Selection.html',message = message,files_complete = data_summary, metric= classification_summary,metric_trim =classification_summ_trim, files = data_summary, plot = " ",visualize=1)

        
        elif request.form["form_name"]=="form3":
            Xaxis = request.form["Xaxis"] 
            Yaxis = request.form["Yaxis"]
            #Xaxis = "f2"
            #Yaxis = "f2"
            infile = open('datastore.json','r')
            path_dict = json.load(infile)
            dataset = pd.read_csv(path_dict['visualize'])
            infile.close()
            data_summary = Summarize_data(dataset)
            #features = dataset.drop('class',axis=1)
            plotfile = os.path.join('static/'+str(time.time()) +'fig1.png' )
            plt.clf()
            if Xaxis==Yaxis:
                figure = sns.distplot(dataset[Xaxis],kde=False)
                fig = figure.get_figure()
                fig.savefig(plotfile) 
            else:
                figure = sns.scatterplot(x=Xaxis,y=Yaxis, hue=data_summary['Y'],data=dataset)
                plt.savefig(plotfile)

           # for filename in glob.glob(os.path.join('static', '*.png')):
            #    os.remove(filename)
            
           
            return render_template('Feature Selection.html',message = " ",files_complete= data_summary,files = data_summary, metric = " ",metric_trim = " ", plot = plotfile, visualize =1)
    
    return render_template('Feature Selection.html',message = " ",files_complete= " ",files = " ", metric = " ",metric_trim = " ", plot = " ",visualize=0)




def Trim_dataset(reorder,cols_required=5):
    infile = open('datastore.json','r')
    path_dict = json.load(infile)
    infile.close()
    if reorder==0:
        path = path_dict['path']
    elif reorder==1:
        path = path_dict['reorder']
    dataset = pd.read_csv(path) 
    dataset.drop(dataset.iloc[:,cols_required:-1],axis=1,inplace=True)
    path = os.path.join("uploads","trimdata.csv")
    dataset.to_csv(path,index= False)
    path_dict.update({"trim":path})
    outfile = open('datastore.json','w')
    json.dump(path_dict,outfile)
    outfile.close() 



def Classify(classifier,trim,reorder):
    infile = open('datastore.json','r')
    path_dict = json.load(infile)
    infile.close()
    if trim == 0:
        if reorder==0:
            path = path_dict['path']
        elif reorder==1:
            path = path_dict['reorder']
    elif trim == 1:
        path = path_dict['trim']
    dataset = pd.read_csv(path) 
    path_dict.update({'visualize':path})  
    outfile = open('datastore.json','w')
    json.dump(path_dict,outfile)
    outfile.close() 
    data_summary = Summarize_data(dataset)
    X_train,X_test,y_train,y_test = train_test_split(data_summary['X'],data_summary['Y'],test_size=0.3,random_state=101)
    if request.form["classifier"]=="logistic":
        accuracy,precision,recall,fscore,specificity,Mcc,g_mean,Conf_matrix = logisticregression(X_train,X_test,y_train,y_test)  
    elif request.form["classifier"]=="random_forest":
        accuracy,precision,recall,fscore,specificity,Mcc,g_mean,Conf_matrix = Random_Forest(X_train,X_test,y_train,y_test,100)
    elif request.form["classifier"]=="naive_bayes":
        accuracy,precision,recall,fscore,specificity,Mcc,g_mean,Conf_matrix = naivebayes(X_train,X_test,y_train,y_test)
    classification_summary = {
                "accuracy":accuracy,
                "precision":precision,
                "recall": recall,
                "fscore": fscore,
                "specificity":specificity,
                "Mcc": Mcc,
                "g_mean": g_mean,
                "Conf_matrix": Conf_matrix
            }  
    return classification_summary,data_summary

def metrics(predictions,y_test,Conf_matrix):
  cm = confusion_matrix(predictions,y_test)
  accuracy = accuracy_score(predictions,y_test)
  precision,recall,fscore,support=score(y_test,predictions,average='weighted')
  tn = cm[1][1]
  fp =  cm[0][1]
  specificity = tn/(tn+fp)
  Mcc = matthews_corrcoef(y_test,predictions)
  #fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=2)
  #Auc = metrics.auc(fpr, tpr)
  g_mean = geometric_mean_score(y_test, predictions, average='weighted')
  return accuracy,precision,recall,fscore,specificity,Mcc,g_mean,Conf_matrix

def logisticregression(X_train,X_test,y_train,y_test):
  #iris = pd.read_csv('dataset.csv')
  #X= iris_data.drop('class',axis=1)
  #y = iris_data['class']
  #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
  from sklearn.linear_model import LogisticRegression
  logmodel = LogisticRegression()
  logmodel.fit(X_train,y_train)
  predictions= logmodel.predict(X_test)
  #accuracy,precision,recall,fscore,specificity = metrics(predictions,y_test)
  Conf_matrix = plot_confusion_matrix(logmodel,X_test,y_test)
  plotfile1 = os.path.join('static/'+str(time.time()) +'fig2.png' )
  plt.savefig(plotfile1)
  return metrics(predictions,y_test,plotfile1)

def naivebayes(X_train,X_test,y_train,y_test):
  #iris = pd.read_csv('dataset.csv')
  #X= iris_data.drop('class',axis=1)
  #y = iris_data['class']
  #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
  from sklearn.naive_bayes import GaussianNB
  naive_bayes = GaussianNB()
  naive_bayes.fit(X_train,y_train)
  predictions = naive_bayes.predict(X_test)
  #accuracy,precision,recall,fscore,specificity = metrics(predictions,y_test)
  Conf_matrix = plot_confusion_matrix(logmodel,X_test,y_test)
  plotfile1 = os.path.join('static/'+str(time.time()) +'fig2.png' )
  plt.savefig(plotfile1)
  return metrics(predictions,y_test,plotfile1)

  
def Random_Forest(X_train,X_test,y_train,y_test,no_decisiontrees=100):
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier(n_estimators=no_decisiontrees)
  model.fit(X_train,y_train)
  predictions = model.predict(X_test)
  Conf_matrix = plot_confusion_matrix(model,X_test,y_test)
  plotfile1 = os.path.join('static/'+str(time.time()) +'fig2.png' )
  plt.savefig(plotfile1)
  return metrics(predictions,y_test,plotfile1)

def Summarize_data(dataset):
    X = dataset.drop(dataset.columns[[-1]],axis=1)
    cols = X.shape[1]
    records = X.shape[0]
    features = X.columns
    Y = dataset.iloc[:,-1]
    data_summary = {
              "X":X,
              "Y":Y,
              "columns": cols,
              "records": records,
              "features":features
    }
    return data_summary


if __name__ == '__main__':
    app.run(debug=True)