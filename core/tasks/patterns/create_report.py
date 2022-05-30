import os
import pandas as pd
import toml

from config.core.config_services import ConfigServices
from tasks.core.task_step import TaskStep


class Task(TaskStep):
    """Create patterns directory

    """

    def run(self, pattern_name, instrument_id, df, score=0.9, params=None):
        """
        Create Pattern

        """
        services_config = ConfigServices.create()
        result = []
        instrument_str = "_".join(instrument_id)

        for name, group in df.groupby("pattern"):
            pos = group[group.profit == 1]
            score = len(pos) / len(group)
            d = {'pattern': name, 'score': score, 'count': len(group)}
            result.append(d)

        # Dataframe
        result_df = pd.DataFrame(result)
        df_dump = result_df[(result_df['score'] > score) & (result_df['count'] > 1)].sort_values(['score', 'count'],
                                                                                                 ascending=False)

        # Create patterns output directory
        patterns_directory = os.path.join(
            services_config.get("MODEL.LOCAL_DIRECTORY"),
            'patterns',
            f'{instrument_str}_{pattern_name.upper()}'
        )
        if not os.path.exists(patterns_directory):
            os.makedirs(patterns_directory)

        # Create output file
        patterns_output_file = os.path.join(patterns_directory, f"{instrument_str.upper()}.parquet")
        df_dump.to_parquet(patterns_output_file, compression='gzip')

        # Create config file
        config = dict()
        config['name'] = pattern_name
        config['parquet'] = os.path.basename(patterns_output_file)
        config['params'] = params
        config_output_file = os.path.join(patterns_directory, f"{instrument_str.upper()}.toml")
        with open(config_output_file, 'w') as fw:
            toml.dump(config, fw)

        # Print parquet
        print(pd.read_parquet(patterns_output_file).head(10))
