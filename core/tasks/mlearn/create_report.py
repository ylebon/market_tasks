import os
import toml
import joblib

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Create model output directory
    """

    def create_output_directory(self, output_directory):
        """Create model output directory"""
        # Set output directory
        model_output_directory = os.path.join(ConfigServices.create().get_value("MODEL.LOCAL_DIRECTORY"), output_directory)
        # Create output directory
        if not os.path.exists(model_output_directory):
            os.makedirs(output_directory)
        return model_output_directory

    def create_model(self, model, model_output_directory, model_name):
        """Create model file"""
        model_file = os.path.join(model_output_directory, "{}.model".format(model_name))
        self.log.info("msg='dump model' model_file={}".format(model_file))
        joblib.dump(model, model_file)

    def create_scaler(self, model_output_directory, model_name, scaler):
        """Create scaler file"""
        scaler_file = os.path.join(model_output_directory, "{}.scaler".format(model_name))
        self.log.info("msg='dump scaler' scaler_file={}".format(scaler_file))
        joblib.dump(scaler, scaler_file)

    def create_config(self, model_output_directory, config):
        """Create config file"""
        config_file = os.path.join(model_output_directory, "model.toml")
        with open(config_file, 'w') as fw:
            toml.dump(config, fw)

    def create_report(self, model_output_directory, report):
        """Create report file"""
        report_txt = os.path.join(model_output_directory, "report.txt")
        with open(report_txt, 'w') as fw:
            fw.write(report)

    def run(self, directory: str, model_name: str, model: str, config: str, output_dir=None, x_scaler=None,
            report=None) -> None:
        """Run task"""
        model_output_directory = self.create_output_directory(directory)
        # Create model
        self.create_model(model, model_output_directory, model_name)
        # Create scaler
        if x_scaler:
            self.create_model(model_output_directory, model_name, x_scaler)
        # Create config
        if config:
            self.create_config(model_output_directory, config)
        # Create report file
        self.create_report(model_output_directory, report)
