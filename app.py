from fastapi import FastAPI 
# from model import CNN 

app = FastAPI() 


@app.get("/")
async def main(): 
    return {"Message":"HELLO THERE"} 
 
@app.get("/predict")
async def makePrediction(): 
     return {"NAME": "JOEY"} 


