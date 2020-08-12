from flask import Flask, render_template,request,session
import os,glob,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score
import sl 

app = Flask(__name__)
app.secret_key = "sravani"
@app.route('/home',methods=["GET","POST"])
def hello_world():
    if request.method =="POST":
        
        if request.form["form_name"]=="form1":
            data_summary = {}
            file = request.files["file"]
            file.save(os.path.join("uploads",file.filename))
            dataset = pd.read_csv(os.path.join("uploads",file.filename))
            X = dataset.drop("class",axis=1)
            cols = X.shape[1]
            records = X.shape[0]
            features = X.columns
            Y = dataset["class"]
            data_summary = {
                "X":X,
                "Y":Y,
                "columns": cols,
                "records": records,
                "features":features
            }
            session['dataset'] = os.path.join("uploads",file.filename)
            #return render_template('home.html',files = data_summary)
            ###rank = filter(request.form["feature"],dataset)
            X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=101)
            if request.form["classifier"]=="logistic":
                accuracy,precision,recall,fscore,specificity,Mcc,g_mean,Conf_matrix = logisticregression(X_train,X_test,y_train,y_test)  
            elif request.form["classifier"]=="random_forest":
                accuracy,precision,recall,fscore,specificity,Mcc,g_mean,Conf_matrix = Random_Forest(X_train,X_test,y_train,y_test,100)
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
            return render_template('home.html',metric= classification_summary,files = data_summary, plot = " ",visualize=1)
        elif request.form["form_name"]=="form2":
            Xaxis = request.form["Xaxis"] 
            Yaxis = request.form["Yaxis"]
            #Xaxis = "f2"
            #Yaxis = "f2"
            dataset = pd.read_csv(session['dataset'])
            #features = dataset.drop('class',axis=1)
            plotfile = os.path.join('static/'+str(time.time()) +'fig1.png' )
            if Xaxis==Yaxis:
                figure = sns.distplot(dataset[Xaxis],kde=False)
                fig = figure.get_figure()
                fig.savefig(plotfile)
            else:
                figure = sns.jointplot(x=Xaxis,y=Yaxis,data=dataset,kind='scatter')
                plt.savefig(plotfile)

           # for filename in glob.glob(os.path.join('static', '*.png')):
            #s    os.remove(filename)
            
           
            return render_template('home.html',files = " ", metric = " ", plot = plotfile, visualize =1)
    
    return render_template('home.html',files = " ", metric = " ", plot = " ",visualize=0)


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
  plotfile = os.path.join('static/'+str(time.time()) +'fig1.png' )
  plt.savefig(plotfile)
  return metrics(predictions,y_test,plotfile)
  
def Random_Forest(X_train,X_test,y_train,y_test,no_decisiontrees=100):
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier(n_estimators=no_decisiontrees)
  model.fit(X_train,y_train)
  predictions = model.predict(X_test)
  Conf_matrix = plot_confusion_matrix(model,X_test,y_test)
  plotfile = os.path.join('static/'+str(time.time()) +'fig1.png' )
  plt.savefig(plotfile)
  return metrics(predictions,y_test,plotfile)
@app.route('/page1')
def about():
    return '<h1>This is my first page!!</h1>'

if __name__ == '__main__':
    app.run(debug=True)
