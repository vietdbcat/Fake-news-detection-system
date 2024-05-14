import json, os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

df = pd.read_csv('data/raw/test.csv')

data = []

print(len(df))

# for index, row in df.iterrows():
#     json_str = row.to_json()
#     json_data = json.loads(json_str)
#     data.append({
#         "title": json_data["title"],
#         "text": json_data["text"],
#         "author": "Anonymous",
#         "label": 0
#     })
    
#     if len(data) == 1000:
#         supabase.table('new').insert(data).execute()
#         data.clear()
#         print("done")
        
# supabase.table('new').insert(data).execute()

