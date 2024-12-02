import pandas as pd
import pickle
import yaml
from sklearn.metrics import accuracy_score
import mlflow
import os

from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/sathvik3103/mlpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="sathvik3103"
os.environ['MLFLOW_TRACKING_PASSWORD']="2b807aeef685745e6d102b595dc47014f7668ea8"

params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']    

    mlflow.set_tracking_uri("https://dagshub.com/sathvik3103/mlpipeline.mlflow")

    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)

    mlflow.log_metric("accuracy",accuracy)
    print(f"Model Accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params['data'],params['model'])

