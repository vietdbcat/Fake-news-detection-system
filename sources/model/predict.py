import sys
sys.path.append("C:\\Users\\BAOVIET\\OneDrive\\Máy tính\\Tai lieu hoc tap\\2024\\FakeNews-Detection-System")

from tranformers import TransformerModel
from sources.preprocessing.process import Process
from configparser import ConfigParser
import pandas as pd

config_path = 'sources/config/config.ini'

config = ConfigParser()
config.read(config_path)

embed_dim = config.getint("Model", "embed_dim")
num_head = config.getint("Model", "num_head")
ff_dim = config.getint("Model", "ff_dim")
vocab_size = config.getint("Model", "vocab_size")
max_len = config.getint("Model", "max_len")
weights = config.get("Model", "weights")

data_path = config.get("Data", "test")

data = {
    "author": "Dr. Maximilian Holland",
    "title": "EVs Take 91.0% Share In Norway — Volvo EX30 Grabs Top Spot",
    "text": "The auto market saw plugin EVs take 91.0% share in Norway in April, roughly flat from 91.1% year on year. BEVs alone took 89.4% share, up from 83.3% YoY. Overall auto volume was 11,241 units, up 25% YoY, a recovery over recent months. April’s best selling BEV…",
}
df = pd.DataFrame.from_dict(data, orient='index').T

model = TransformerModel(max_len, vocab_size, embed_dim, num_head, ff_dim)
model.load_weights(weights)

preprocess = Process(vocab_size, max_len)
X, _ = preprocess.process_test_data(df)

label = model.predict(X)
print(label[0][0])