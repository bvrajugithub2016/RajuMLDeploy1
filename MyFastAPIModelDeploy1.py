#subpart-1: imports
import uvicorn
from fastapi import FastAPI
from MyBankNote import BankNote

import pickle
import pandas as pd

#subpart-2: Create the FastAPI object
appObj = FastAPI()

pickle_input_file = open("RandomForestClassifier1.obj", "rb")

rfClassifier = pickle.load(pickle_input_file)

#subpart-3: General Routing
#creating route1 (also known as API)
@appObj.get('/')
def index():
    return {'message', 'Hello BVRaju, Welcome!'}

#route2
@appObj.get("/Welcome")
def get_name(name: str):
    return {'message', f'Hello {name}, Welcome to datapro!'}


#subpart-4: Routing specific to our Prediction Model
@appObj.post('/predict')
def predict_bankNote_fakeness(data: BankNote):
    myDict = data.dict()
    variance = myDict['variance'] 
    skewness = myDict['skewness'] 
    curtosis = myDict['curtosis'] 
    entropy = myDict['entropy']
    
    predictions = rfClassifier.predict([[variance, skewness, curtosis, entropy]])
    print("predictions have been done")
    #remember that we are dealing with only one record
    
    if(predictions[0] > 0.5):
        predictionResult = 'Fake Note'
    else:
        predictionResult = 'Genuine Note'
    
    return { 'predictionResult' : predictionResult}
    #returning a dictionary

#subpart-5: Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(appObj, host='127.0.0.1', port=8000)
    
    