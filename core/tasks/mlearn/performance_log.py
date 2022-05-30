import os

from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Performance log

    """

    def run(self, estimator_name, estimator_params, performance, output_directory):
        report_txt = os.path.join(output_directory, "performance.txt")
        report_data = f"""{estimator_name} - {estimator_params}
        {performance.get('class_report')}
        """
        with open(report_txt, 'a') as fw:
            fw.write(report_data)
            fw.write("-"*10)
            fw.write("\n")
