import os
import re
from glob import glob

import pandas as pd

from tasks.core.task_step import TaskStep
from config.core.config_services import ConfigServices

# from bokeh.layouts import layout
# We're not demo-ing Tabs or DataTable, as they are available in other demos
# from bokeh.models import widgets
# from bokeh.plotting import show


class Task(TaskStep):
    """
    Create data map visualization

    """

    def run(self, input_directory=None, parquet_filter=None):
        """
        Sync amazon s3 data

        """
        input_directory = ConfigServices.create().get_value('PARQUET.LOCAL_DIRECTORY')

        # Create input directory
        input_directory = input_directory or self.__class__.input_directory
        input_directory = os.path.dirname(input_directory)
        input_directory = os.path.abspath(os.path.expanduser(input_directory))
        self.log.info("msg='reading parquet directory' directory={}".format(input_directory))

        # create map
        search_pattern = os.path.join(input_directory, "parquet/exchange=*/symbol=*/date=*/*.parquet")
        parquet_files = glob(search_pattern)
        data = list()

        search_pattern = f'parquet/exchange=(.*)/symbol=(.*)/date=(.*)/.*.parquet'

        for parquet_file in parquet_files:
            d = dict()
            exchange, symbol, date = re.search(search_pattern, parquet_file).groups()
            d['file'] = parquet_file
            d['exchange'] = exchange
            d['symbol'] = symbol
            d['instrument'] = exchange + "_" + symbol
            d['date'] = date
            data.append(d)

        # pandas dataframe
        dataframe = pd.DataFrame(data)
        dataframe['date'] = pd.to_datetime(dataframe['date'], format="%Y%m%d")

        # visualize
        self.visualize(dataframe)

    def visualize(self, dataframe):
        """
        Visualize data

        """
        import matplotlib

        matplotlib.use('TkAgg')
        data = list()

        for group_name, group_df in dataframe.groupby(by='date'):
            d = {'date': group_name}
            for instrument in group_df.instrument:
                d[instrument] = 'OK'
            data.append(d)

        def highlight_missing(s):
            return ['background-color: white' if v == 'OK' else 'background-color: red' for v in s]

        df = pd.DataFrame(data)
        divs = list()
        for group_name, group_df in df.groupby(df['date'].dt.strftime('%U')):
            df_style = group_df.set_index('date').style.apply(highlight_missing)
            text = "<h2> Week: {0}</h2>".format(group_name)
            text += df_style.render()
            divs.append(widgets.Div(text=text))
        l = layout(divs)
        show(l)


if __name__ == '__main__':
    from logbook import StreamHandler, NestedSetup
    import sys
    from tasks.core.tasks_factory import TasksFactory

    # subscribe to instruments
    format_string = (
        '[{record.time:%Y-%m-%d %H:%M:%S.%f%z}] '
        '{record.level_name: <5} - {record.channel: <15}: {record.message}'
    )

    handlers = [
        StreamHandler(sys.stdout, level='INFO', format_string=format_string, bubble=True),
    ]

    with NestedSetup(handlers):
        tasks = TasksFactory.create()
        tasks.parquet.create_map.run(input_directory=r"/Users/madazone/Desktop/parquet_s3/",
                                     parquet_filter=dict(exchange="oanda"))
