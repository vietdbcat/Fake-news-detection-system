import sys
sys.path.append("C:\\Users\\BAOVIET\\OneDrive\\Máy tính\\Tai lieu hoc tap\\2024\\FakeNews-Detection-System")

from tranformers import TransformerModel
from sources.preprocessing.process import Process
from configparser import ConfigParser
from sklearn.model_selection import train_test_split

config_path = 'sources/config/config.ini'

config = ConfigParser()
config.read(config_path)

embed_dim = config.getint("Model", "embed_dim")
num_head = config.getint("Model", "num_head")
ff_dim = config.getint("Model", "ff_dim")
vocab_size = config.getint("Model", "vocab_size")
max_len = config.getint("Model", "max_len")
weights = config.get("Model", "weights")

data_path = config.get("Data", "train")

model = TransformerModel(max_len, vocab_size, embed_dim, num_head, ff_dim)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

preprocess = Process(vocab_size, max_len)
X, y = preprocess.process_train_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size = 32)
model.save(weights)