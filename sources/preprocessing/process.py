import pandas as pd

class Process:
    def __init__(self, data : str):
        self.data = pd.read_csv(data)
        
    def call(self):
        return self.data