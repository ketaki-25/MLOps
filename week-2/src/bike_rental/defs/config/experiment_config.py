import json
import yaml


class ExperimentConfigResource:

    def __init__(
        self,
        yaml_path,
        active_json_path,
    ):
        self.yaml_path = yaml_path
        self.active_json_path = active_json_path

    def get_active_experiment(self):

        with open(self.active_json_path) as f:
            active = json.load(f)

        experiment_name = active["experiment"]

        with open(self.yaml_path) as f:
            experiments = yaml.safe_load(f)

        #TODO:
        # make sure to validate the values being passed in the yaml file. Cross check if the column names exist.

        return experiments["experiments"][experiment_name]