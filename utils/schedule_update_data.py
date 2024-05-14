import json, os
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
import requests
from newsapi import NewsApiClient
from newspaper import Article

load_dotenv()
FOLDER_PATH = os.environ.get("FOLDER_PATH")
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

# newsapi = NewsApiClient(api_key=os.environ.get("NEWS_APIKEY"))

# sources = newsapi.get_sources(language='en')

# top_headlines = newsapi.get_top_headlines(
#     q='new',
#     # sources='bbc-news,cnn,fox-news', 
#     language='en',
#     page_size=100
# )

# data = []

# for new in top_headlines['articles']:
#     content = new['content']
#     if content == None and new['description'] != None:
#         content = new['description']
#     if content == None:
#         continue

#     try:
#         resp = requests.get(new['url'], timeout=5)
#         article = Article('')
#         article.download(input_html=resp.text)
#         article.parse()
#         data.append({
#             "title": new['title'],
#             "author": new['author'],
#             "text": article.text if article.text and len(article.text) > len(content) else content,
#             "label": 1
#         })
#     except:
#         data.append({
#             "title": new['title'],
#             "author": new['author'],
#             "text": content,
#             "label": 1
#         })
        
# supabase.table('new').insert(data).execute()
print("Update data successfully!")
data = []
cluster_length = 1000
cluster_index = 1
flag = 1

while flag:
    result = supabase.table('new').select("*") \
        .order("created_at", desc=True)\
        .offset((cluster_index - 1) * cluster_length)\
        .limit(cluster_length)\
        .execute()
    cluster_index = cluster_index + 1
    data = data + result.data
    print(f"Get {len(data)} rows successfully!")
    if (result.data == []):
        flag = 0
print("Get data successfully!")
    
dataTrain = []
dataTest = []
for i in range(len(data)):
    dataTrain.append((
        i + 1,
        data[i]['title'],
        data[i]['author'],
        data[i]['text'],
        1 if data[i]['label'] == True else 0
    ))
    dataTest.append((
        i + 1,
        data[i]['title'],
        data[i]['author'],
        data[i]['text'],
        1 if data[i]['label'] == True else 0
    ))
df = pd.DataFrame(dataTrain, columns=['id', 'title', 'author', 'text', 'label'])
df.to_csv(f'{FOLDER_PATH}/data/raw/train.csv', index=False)
df = pd.DataFrame(dataTest, columns=['id', 'title', 'author', 'text', 'label'])
df.to_csv(f'{FOLDER_PATH}/data/raw/test.csv', index=False)