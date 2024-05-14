import json, os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

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
    print(len(data))
    if (result.data == []):
        flag = 0
    
dataFormat = []
for i in range(len(data)):
    dataFormat.append((
        i + 1,
        data[i]['title'],
        data[i]['author'],
        data[i]['text'],
        1 if data[i]['label'] == True else 0
    ))
print(dataFormat[0])
df = pd.DataFrame(data, columns=['id', 'title', 'author', 'text', 'label'])
df.to_csv('data/raw/train.csv', index=False)