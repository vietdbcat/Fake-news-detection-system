import sys, os
from dotenv import load_dotenv 
load_dotenv()
sys.path.append(os.environ.get("FOLDER_PATH"))

import numpy as np
from .model import TransformerModel
from .process import Process
from configparser import ConfigParser
import pandas as pd

config_path = 'utils/config.ini'

config = ConfigParser()
config.read(config_path)

embed_dim = config.getint("Model", "embed_dim")
num_head = config.getint("Model", "num_head")
ff_dim = config.getint("Model", "ff_dim")
vocab_size = config.getint("Model", "vocab_size")
max_len = config.getint("Model", "max_len")
weights = config.get("Model", "weights")


def predict(data):
    model = TransformerModel(max_len, vocab_size, embed_dim, num_head, ff_dim)
    model.load_weights(weights)
    preprocess = Process(vocab_size, max_len)
    df = pd.DataFrame.from_dict(data, orient='index').T
    X, _ = preprocess.process_test_data(df)    
    label = model.predict(X)
    result = float(np.array(label[0]))
    return result