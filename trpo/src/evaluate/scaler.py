import pickle
import numpy as np


class Scaler(object):
    def __init__(self, model_path):
        scaler_file = open(model_path + "/info/scalar.pkl", "rb")
        scaler_data = pickle.load(scaler_file)
        scaler_file.close()
        self.vars = scaler_data['vars']
        self.means = scaler_data['means']

    def get(self):
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means
