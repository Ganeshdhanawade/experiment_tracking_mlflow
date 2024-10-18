import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt 
import seaborn as sns
import mlflow.sklearn


mlflow.set_experiment("water_autologing_mlflow")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.read_csv('/home/ganesh/mlflow/exp_tracking_mlflow_water_potability/data/water_potability.csv')

from sklearn.model_selection import train_test_split
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

from sklearn.ensemble import GradientBoostingClassifier
import pickle
X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

n_estimators = 500

#mlflow part for tracking the parameters

#autologging of the all experiments
mlflow.autolog()  #autologging all the experiment without writing mannualy

with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    #save
    pickle.dump(clf,open('model.pkl','wb'))


    ### for the prediction
    x_test=test_processed_data.iloc[:,0:-1].values
    y_test=test_processed_data.iloc[:,-1].values


    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    model = pickle.load(open('model.pkl','rb'))

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_scores =f1_score(y_test,y_pred)

    print("acc", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1_scores", f1_scores)