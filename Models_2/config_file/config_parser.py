import yaml

class Data:

    def extractData(self):
        with open("config_file/parameters.yaml", 'r') as stream:
            try:
                p = yaml.safe_load(stream)
                self.trials = p["trials"]
                self.epochs = p["epochs"]
                self.balanced = p["balanced"]
                self.repetition = p["repetition"]
            except yaml.YAMLError as exc:
                print(exc)