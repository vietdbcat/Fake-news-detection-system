import sys
sys.path.append("C:\\Users\\BAOVIET\\OneDrive\\Máy tính\\Tai lieu hoc tap\\2024\\FakeNews-Detection-System")

from tranformers import TransformerModel
from sources.preprocessing.process import Process
from configparser import ConfigParser

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

model = TransformerModel(max_len, vocab_size, embed_dim, num_head, ff_dim)
model.load_weights(weights)

preprocess = Process(vocab_size, max_len)
X, _ = preprocess.process_test_data(data_path)

label = model.predict(X)
print(label)