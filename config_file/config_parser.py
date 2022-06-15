import yaml
import sys
# sys.path.append('../')
class Data:

    def extractData(self):
        with open("../config_file/parameters.yaml", 'r') as stream:
            try:
                p = yaml.safe_load(stream)
                self.trials = p["trials"]
                self.epochs = p["epochs"]
                self.balanced = p["balanced"]
                self.repetition = p["repetition"]
                # self.interpolation = p["interpolation"]
                # self.regularization = p["regularization"]
                # self.indicator = p["indicator"]
            except yaml.YAMLError as exc:
                print(exc)