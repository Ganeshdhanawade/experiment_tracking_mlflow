import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import mlflow

import matplotlib.pylab as plt 
import seaborn as sns
import mlflow.sklearn


mlflow.set_experiment("water_hyperparameter_track")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.read_csv('/home/ganesh/mlflow/exp_tracking_mlflow_water_potability/data/water_potability.csv')

train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df

## Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data =fill_missing_with_median(test_data)

import pickle
X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

## define the model and parameter distribution for randomizedsearchcv
rf=RandomForestClassifier(random_state=42)
param_dict = { 
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30, 40]
}

#Perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(estimator=rf,param_distributions=param_dict, cv=5, n_jobs= 32, verbose=2, random_state=4)

##tracking by using mlflow
with mlflow.start_run():
    random_search.fit(X_train, y_train)

    #Print the best hyperparameter found by RandomizedSearchCV
    print("Best parameter found: ",random_search.best_params_)

    #train the model with the best parameters
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train,y_train)


    pickle.dump(best_rf,open('model.pkl','wb'))


    ### for the prediction
    x_test=test_processed_data.iloc[:,0:-1].values
    y_test=test_processed_data.iloc[:,-1].values

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)


    model = pickle.load(open('model.pkl','rb'))

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_scores =f1_score(y_test,y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("fl_scores", f1_scores)

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df,"train") #track the dataset on mlflow
    mlflow.log_input(train_df,"test")
        
    mlflow.log_artifact(__file__) #track the code using mlflow

    mlflow.sklearn.log_model(random_search.best_estimator_,"Best Model")

    print("acc", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1_scores", f1_scores)