#!/usr/bin/env python
# coding: utf-8

# In[14]:


from fastapi import FastAPI, Path
from pydantic import BaseModel
from typing import Optional
from joblib import load
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# In[15]:


# Importez du modèle 

model = load('lgbm_w.joblib')

app = FastAPI()
# scale de donnees 
scaler = MinMaxScaler(feature_range = (0, 1))

df=pd.read_csv('./test.csv')
list_clients=df['SK_ID_CURR']
print(list_clients)

class ClientInput(BaseModel):
    client_id: int

class PredictionOutput(BaseModel):
    client_id: int
    predicted_class: int

@app.get("/predict/{client_id}")
async def predict_class(client_id: int = Path(..., title="ID du client")):
    # recuperation des caracteristiques du dataframe
    if client_id in list_clients:
        features_for_client_id = get_features_for_client_id(client_id)
        if features_for_client_id is not None:
            predicted_class = int(model.predict([features_for_client_id])[0] > 0.681)  
        
            # la classe de sortie
            output = PredictionOutput(client_id=client_id, predicted_class=predicted_class)
            return output
        else:
            return {"error": "Client non trouvé"}
    else:
        print('client non trouvé') 
# Fonction pour récupérer les caractéristiques du client
def get_features_for_client_id(client_id):
    df=pd.read_csv('./test.csv')
    
    # recherche du client 
    client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])
    if client_data.shape[0]!=0:
        scaler.fit(client_data)
        scaled_client_data = scaler.transform(client_data)
   
    return client_data.values[0] if not client_data.empty else None

 
@app.get('/')
def index():
    return "index"


# In[ ]:




