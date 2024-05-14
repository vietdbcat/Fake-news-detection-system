from datetime import datetime
import select
import sys, os
from typing import *
from dotenv import load_dotenv
from fastapi import *
from pydantic import Json
from pydantic import BaseModel
from utils.predict import predict
from supabase import create_client, Client

load_dotenv()

sys.path.append(os.environ.get("FOLDER_PATH"))

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

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
    check_new = supabase\
        .table('new')\
        .select("*")\
        .eq('title', new.title)\
        .eq('text', new.text)\
        .execute()
    
    if len(check_new.data) > 0:
        return check_new.data[0]
    
    result = predict({
        "author": new.author,
        "title": new.title,
        "text": new.text
    })
    
    data, count = supabase.table('new').insert({
            "author": new.author,
            "title": new.title,
            "text": new.text,
            "label": 1 if result > 0.5 else 0
        }).execute()
    
    return data

@app.get("/get_news")
def get_news(
    id: str = Query(None, description="Filter news by id"),
    page: int = Query(1, description="Filter news by page"),
    author: str = Query(None, description="Filter news by author"),
    title: str = Query(None, description="Filter news by title"),
    fromDate: str = Query(None, description="Filter news by from date"),
    endDate: str = Query(None, description="Filter news by end date")
):
    query = supabase.table('new').select("*")
    
    if id:
        query = query.eq("id", f"{id}")
    if author:
        query = query.ilike("author", f"%{author}%")
    if title:
        query = query.ilike("title", f"%{title}%")
    if fromDate:
        fromDate = datetime.strptime(fromDate, "%Y-%m-%d")
        query = query.gte("created_at", fromDate.isoformat())
    if endDate:
        endDate = datetime.strptime(endDate, "%Y-%m-%d")
        query = query.lte("created_at", endDate.isoformat())   
        
    result = query\
        .order("created_at", desc=True)\
        .offset((page - 1) * 9)\
        .limit(9)\
        .execute()
        
    print(len(result.data))
    return result.data



