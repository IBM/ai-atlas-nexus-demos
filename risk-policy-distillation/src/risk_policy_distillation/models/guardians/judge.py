import ast
import configparser


class Judge:
    def __init__(self, config):
        self.process_config(config)

    def process_config(self, c):
        config = configparser.ConfigParser()
        config.read_dict({"guardian": c})

        self.task = config.get("guardian", "task")
        self.output_labels = ast.literal_eval(config.get("guardian", "output_labels"))
        self.criterion = config.get("guardian", "criterion")
        self.definition = config.get("guardian", "criterion_definition")
        self.labels = ast.literal_eval(config.get("guardian", "labels"))
        self.label_names = ast.literal_eval(config.get("guardian", "label_names"))

        self.n_labels = len(self.labels)

    def ask_guardian(self, message):
        pass

    def predict_proba(self, message):
        pass
