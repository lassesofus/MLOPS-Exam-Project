from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Git merge problems 
# How to upgit load weights to docker image


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}