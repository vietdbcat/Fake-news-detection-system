# import sys
# sys.path.append("C:\\Users\\BAOVIET\\OneDrive\\Máy tính\\Tai lieu hoc tap\\2024\\FakeNews-Detection-System")

# from sources.model.tranformers import TransformerModel
# from sources.preprocessing.process import Process
# from configparser import ConfigParser
# import pandas as pd

# config_path = 'sources/config/config.ini'

# config = ConfigParser()
# config.read(config_path)

# embed_dim = config.getint("Model", "embed_dim")
# num_head = config.getint("Model", "num_head")
# ff_dim = config.getint("Model", "ff_dim")
# vocab_size = config.getint("Model", "vocab_size")
# max_len = config.getint("Model", "max_len")
# weights = config.get("Model", "weights")

# data_path = config.get("Data", "test")

# data = {
#     "author": "Dr. Maximilian Holland",
#     "title": "EVs Take 91.0% Share In Norway — Volvo EX30 Grabs Top Spot",
#     "text": "The auto market saw plugin EVs take 91.0% share in Norway in April, roughly flat from 91.1% year on year. BEVs alone took 89.4% share, up from 83.3% YoY. Overall auto volume was 11,241 units, up 25% YoY, a recovery over recent months. April’s best selling BEV…",
# }
# df = pd.DataFrame.from_dict(data, orient='index').T

import pandas as pd

# df = pd.read_csv("data/raw/train.csv")
# label = df["label"]
# title = df["title"]
# text = df["text"]

# print(f"label: {label[1]}")
# print(f"title: {title[1]}")
# print(f"content: {text[1]}")

# for i, t in enumerate(title):
#     print(f"title[{i}]: {t}")
#     print(f"---> Label: {label[i]}")
#     print()
#     if i > 10: break

from sources.preprocessing.process import Process

df = pd.read_csv("data/raw/test.csv")
# print(df["title"][22])
preprocess = Process(5000, 20)
d, _ = preprocess.process_test_data(df)
print(len(d))