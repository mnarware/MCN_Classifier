print("Initializing Algorithm ....")
import pandas as pd
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
print("Initialized successfully !")

def rfc(X,y):

        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
        model_rfc=RandomForestClassifier()
        print('Data is processing into RandomForestClassifier !!!')
        model_rfc.fit(xtrain,ytrain)
        print('Data processing is finished!!!')
        pred=model_rfc.predict(xtrain)
        score=accuracy_score(ytrain,pred)
        print('Your accuracy score in percent :',round((score*100),2),'%')
        
        
def adboost(X,y):
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
        model_rfc=AdaBoostClassifier()
        print('Data is processing into AdaBoostClassifier !!!')
        model_rfc.fit(xtrain,ytrain)
        print('Data processing is finished!!!')
        pred=model_rfc.predict(xtrain)
        score=accuracy_score(ytrain,pred)
        print('Your accuracy score in percent :',round((score*100),2),'%')
        
    
def bag(X,y):
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
        model_rfc=BaggingClassifier()
        print('Data is processing into BaggingClassifier !!!')
        model_rfc.fit(xtrain,ytrain)
        print('Data processing is finished!!!')
        pred=model_rfc.predict(xtrain)
        score=accuracy_score(ytrain,pred)
        print('Your accuracy score in percent :',round((score*100),2),'%')

def dtc(X,y):
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
        model_rfc=DecisionTreeClassifier()
        print('Data is processing into DecisionTreeClassifier !!!')
        model_rfc.fit(xtrain,ytrain)
        print('Data processing is finished!!!')
        pred=model_rfc.predict(xtrain)
        score=accuracy_score(ytrain,pred)
        print('Your accuracy score in percent :',round((score*100),2),'%')

        
def logistic(X,y):
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
        model_rfc=LogisticRegression()
        print('Data is processing into LogisticRegression !!!')
        model_rfc.fit(xtrain,ytrain)
        print('Data processing is finished!!!')
        pred=model_rfc.predict(xtrain)
        score=accuracy_score(ytrain,pred)
        print('Your accuracy score in percent :',round((score*100),2),'%')
        
        
def knc(X,y):
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
        model_rfc=KNeighborsClassifier(n_neighbors=10)
        print('Data is processing into KNeighborsClassifier !!!')
        model_rfc.fit(xtrain,ytrain)
        print('Data processing is finished!!!')
        pred=model_rfc.predict(xtrain)
        score=accuracy_score(ytrain,pred)
        print('Your accuracy score in percent :',round((score*100),2),'%')

def all_algo_score(X,y):

        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)

        model_rfc=RandomForestClassifier()
        model_rfc.fit(xtrain,ytrain)
        pred=model_rfc.predict(xtrain)
        rfc_score=accuracy_score(ytrain,pred)


        model_ad=AdaBoostClassifier()
        model_ad.fit(xtrain,ytrain)
        pred=model_ad.predict(xtrain)
        adb_score=accuracy_score(ytrain,pred)


        model_bag=BaggingClassifier()
        model_bag.fit(xtrain,ytrain)
        pred=model_bag.predict(xtrain)
        bag_score=accuracy_score(ytrain,pred)

        model_dtc=DecisionTreeClassifier()
        model_dtc.fit(xtrain,ytrain)
        pred=model_dtc.predict(xtrain)
        dtc_score=accuracy_score(ytrain,pred)


        model_lg=LogisticRegression()
        model_lg.fit(xtrain,ytrain)
        pred=model_lg.predict(xtrain)
        lg_score=accuracy_score(ytrain,pred)


        model_kn=KNeighborsClassifier(n_neighbors=10)
        model_kn.fit(xtrain,ytrain)
        pred=model_kn.predict(xtrain)
        score_kn=accuracy_score(ytrain,pred)

        print('| ===========================================')
        print('| Algorithms              | Accuracy Score')
        print('| ===========================================')
        print('| RandomForestClassifier  |',round(rfc_score*100,2),'%')
        print('| AdaBoostClassifier      |',round(adb_score*100,2),'%')
        print('| BaggingClassifier       |',round(bag_score*100,2),'%')
        print('| DecisionTreeClassifier  |',round(dtc_score*100,2),'%')
        print('| LogisticRegression      |',round(lg_score*100,2),'%')
        print('| KNeighborsClassifier    |',round(score_kn*100,2),'%')
        print('|============================================')

        
def mychoice(X,y):
    ch=0

    while(True):
            print('1. DecisionTreeClassifier Algorithm')
            print('2. RandomForestClassifier Algorithm ')
            print('3. LogisticRegression Algorithm')
            print('4. AdaBoostClassifier Algorithm ')
            print('5. BaggingClassifier Algorithm ')
            print('6. KNeighborsClassifier Algorithm ')
            print('7. Accuracy score of all Algorithm ')
            print('8 Exit ')
            time.sleep(1)
            ch=int(input('Please enter your above choice for e.g 1 for DecisionTree  : '))

            if ch==1:
                print('=====================================================================')
                dtc(X,y)
                print('=====================================================================')


            elif ch==2:
                print('=====================================================================')
                rfc(X,y)
                print('=====================================================================')



            elif ch==3:
                print('=====================================================================')
                logistic(X,y)
                print('=====================================================================')


            elif ch==4:
                print('=====================================================================')
                adboost(X,y)
                print('=====================================================================')

            elif ch==5:
                print('=====================================================================')
                bag(X,y)
                print('=====================================================================')

            elif ch==6:
                print('=====================================================================')
                knc(X,y)
                print('=====================================================================')

            elif ch==7:
                print('=====================================================================')
                all_algo_score(X,y)
                print('=====================================================================')

            elif ch==8:
                print('============= Program is Stopped ================')
                break
            else:
                print('============= Invalid Choice !!! ================')
                break
