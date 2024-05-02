
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import export_graphviz
#from IPython import display
from Allcodefiles.Dataread import read_csv
from Allcodefiles.DataHandling import missing_values
from Allcodefiles.DataHandling import texthandling
from Allcodefiles.DataHandling import categorical_cols_train
from model.Model import train_and_save_model
from Allcodefiles.Exceptionhandling import handle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
from Allcodefiles.Exceptionhandling import handle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from Allcodefiles.Dataread import read_csv
from Allcodefiles.Dataread import read_csv
from Allcodefiles.DataHandling import missing_values
from Allcodefiles.DataHandling import texthandling
from Allcodefiles.DataHandling import categorical_cols_test
#from Allcodefiles.Predict import load_model_predict
from Allcodefiles.Exceptionhandling import handle
main = tkinter.Tk()
main.title("Fake Job Detection") #designing main screen
main.geometry("1300x1200")

global filename
global cls
global X, Y, X_train, X_test, y_train, y_test
global random_acc # all global variables names define in above lines
global clean
global attack
global total


def traintest(train):     #method to generate test and train data from dataset
    fake_job_postings = pd.read_csv('dataset/fake_job_postings_cleaned1.csv')
    X = fake_job_postings[['telecommuting', 'ratio', 'text', 'character_count']]
    Y = fake_job_postings['fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=53)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel(): #method to read dataset values which contains all five features data
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = traintest(train)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")



def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");



def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(50):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
from sklearn.ensemble import RandomForestClassifier
        randomforest=RandomForestClassifier(n_estimators=5)
        randomforest.fit(tfidf_train,y_train)
        y_pred_rf=randomforest.predict(tfidf_test)
        #print(confusion_matrix(y_test,y_pred_rf))
        #print(accuracy_score(y_test, y_pred_rf)*100)
        #print(classification_report(y_test,y_pred_rf))


def runRandomForest():
    
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    X_train_num = X_train[['telecommuting', 'ratio', 'character_count']]
    X_test_num = X_test[['telecommuting', 'ratio', 'character_count']]
    count_vectorizer = CountVectorizer(stop_words='english')
    count_train = count_vectorizer.fit_transform(X_train.text.values)
    count_test = count_vectorizer.transform(X_test.text.values)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=1)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train.text)
    tfidf_test = tfidf_vectorizer.transform(X_test.text)
    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
    cls = MultinomialNB()
    cls.fit(count_train, y_train)
    pred = cls.predict(count_test)
    nv=metrics.accuracy_score(y_test, pred)*100
    print(nv)
    pred1 = cls.predict(count_test[0:10])
    print( X_test[0:10])
    print(pred1)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(count_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Accuracy')
    #str_tree = export_graphviz(cls, out_file=None, feature_names=headers,filled=True, special_characters=True, rotate=True, precision=0.6)
    #display.display(str_tree)
                


def predicts():
    global clean
    global attack
    global total
    clean = 0;
    attack = 0;
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    data = read_csv('dataset/test.csv')

    (data.pipe(missing_values).pipe(texthandling)
             .pipe(categorical_cols_test).pipe(load_model_predict))     
    
     


def load_model_predict(data):
    try:
        X_test = data.drop('fraudulent',axis = 1)
        y_test = data['fraudulent']

        scaler = pickle.load( open( "model/scaler.p", "rb" ) )
        X_test = scaler.transform(X_test)

        filename = 'model/finalized_model.p'
        model = pickle.load(open(filename, 'rb'))

        y_pred = model.predict(X_test)
        score_and_save(y_pred)        
    except Exception as e:
        handle('prediction process')



def score_and_save(y_pred):
    try:
        data = read_csv('dataset/test.csv')

        y_test = data['fraudulent']
        cm = confusion_matrix(y_test, y_pred)
        print("\n"+"SCORES")
        print("confusion matrix")
        print(cm)
        print('F1-Score'+' = '+str(round(f1_score(y_test, y_pred),4)))
        print('Precision'+' = '+str(round(precision_score(y_test, y_pred),4)))
        print('Recall'+' = '+str(round(recall_score(y_test, y_pred),4)))
        print('Accuracy'+' = '+str(round(accuracy_score(y_test,y_pred),4)))
      
        data['fraud_prediction'] = y_pred

        data.to_csv('predictionoutput/testsetprediction.csv')
        text.insert(END,"X=%s, Predicted = %s" % (str(round(f1_score(y_test, y_pred),4)), 'F1-Score')+"\n\n")
        text.insert(END,"X=%s, Predicted = %s" % (str(round(precision_score(y_test, y_pred),4)), 'Precision')+"\n\n")
        text.insert(END,"X=%s, Predicted = %s" % (str(round(recall_score(y_test, y_pred),4)), 'Recall')+"\n\n")
        text.insert(END,"X=%s, Predicted = %s" % (str(round(accuracy_score(y_test,y_pred),4)), 'Accuracy')+"\n\n")
    except Exception as e:
        handle('scoring and saving process')



font = ('times', 16, 'bold')
title = Label(main, text='Fake Job Detection Using Random Forest Classifier')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Fake Job Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Generate Train & Test Model", command=generateModel)
modelButton.place(x=350,y=550)
modelButton.config(font=font1) 

runrandomButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomButton.place(x=650,y=550)
runrandomButton.config(font=font1) 

predictButton = Button(main, text="Detect fakejob From Test Data", command=predicts)
predictButton.place(x=50,y=600)
predictButton.config(font=font1) 
 



exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=880,y=600)
exitButton.config(font=font1) 

main.config(bg='LightSkyBlue')
main.mainloop()
