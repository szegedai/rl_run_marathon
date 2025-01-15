import pickle
import numpy as np
import os
import csv


class Scaler(object):
    def __init__(self, obs_dim, model_path=None, load_prev=False):
        if(load_prev):
            scaler_file = open(model_path + "/info/scalar.pkl", "rb")
            scaler_data = pickle.load(scaler_file)
            scaler_file.close()
            self.vars = scaler_data['vars']
            self.means = scaler_data['means']
            self.m = scaler_data['m']
            self.first_pass = False
        else:
            self.vars = np.zeros(obs_dim)
            self.means = np.zeros(obs_dim)
            self.m = 0
            self.first_pass = True

    def update(self, x):
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    def get(self):
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


class Logger(object):
    def __init__(self, logname, now):
        path = os.path.join('..', 'log-files', logname, 'csvs', now)
        os.makedirs(path)
        path = os.path.join(path, 'log.csv')

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None

    def write(self, display=True):
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames, lineterminator='\n')
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                               log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        self.log_entry.update(items)

    def close(self):
        self.f.close()
