import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt 
import seaborn as sns
import mlflow.sklearn


mlflow.set_experiment("water_exp2")
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

with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    #save
    pickle.dump(clf,open('model.pkl','wb'))


    ### for the prediction
    x_test=test_processed_data.iloc[:,0:-1].values
    y_test=test_processed_data.iloc[:,-1].values

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    model = pickle.load(open('model.pkl','rb'))

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_scores =f1_score(y_test,y_pred)

    #track with mlflow
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_scores", f1_scores)

    mlflow.log_param("n_estimators",n_estimators)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True)
    plt.xlabel("predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matric")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")  #track confusion matrix

    mlflow.sklearn.log_model(clf,"GradientBoostingClassifier") #track the model
    
    mlflow.log_artifact(__file__) #track the code using mlflow

    mlflow.log_input(train_df,"train") #track the dataset on mlflow
    mlflow.log_input(train_df,"test")


    mlflow.set_tag("author","datathinkers") #add tags in mlflow to search keywords
    mlflow.set_tag("model","GB") #same work as the above to serch the experiment
    
    print("acc", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1_scores", f1_scores)