import sys
sys.path.append("/home/huy31/Projects/KDLKP/Fake-news-detection-system")

from typing import *

from fastapi import *
from pydantic import Json
from pydantic import BaseModel
from utils.predict import predict

app = FastAPI()

data = {
    "author": "Dr. Maximilian Holland",
    "title": "US Officials See No Link Between Trump and Russia",
    "text": "The auto market saw plugin EVs take 91.0% share in Norway in April, roughly flat from 91.1% year on year. BEVs alone took 89.4% share, up from 83.3% YoY. Overall auto volume was 11,241 units, up 25% YoY, a recovery over recent months. April’s best selling BEV…"
}

class New(BaseModel):
    author: str
    title: str
    text: str

@app.post("/check-health")
def check_health(
    request: Request
):
    return {"Hello World"}

@app.post("/push_new")
def push_new(
    new: New
):
    result = predict({
        "author": new.author,
        "title": new.title,
        "text": new.text
    })
    return result


