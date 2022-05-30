"""Tasks launcher.

Usage:
  tasks_launcher.py backtesting (--signaler=<signaler>) (--instrument=<instrument>) (--past=<past>) [--env=<env>] [--loglevel=<loglevel>]
  tasks_launcher.py features_creator (--instrument=<instrument>) [--feature=<feature>] (--date=<date>) [--env=<env>] [--loglevel=<loglevel>]
  tasks_launcher.py features_importance (--instrument=<instrument>) (--target=<target>) [--env=<env>] [--feature=<feature>] [--loglevel=<loglevel>]
  tasks_launcher.py backup (--instrument=<instrument>) (--start-date=<start-date>) (--end-date=<end-date>) [--loglevel=<loglevel>]
  tasks_launcher.py training_tpot (--instrument=<instrument>) (--target=<target>) [--estimator=<estimator>] [--env=<env>] [--feature=<feature>] [--loglevel=<loglevel>]
  tasks_launcher.py training_automl (--instrument=<instrument>) (--target=<target>) [--env=<env>] [--feature=<feature>] [--loglevel=<loglevel>]
  tasks_launcher.py training_autokeras (--instrument=<instrument>) (--target=<target>) [--env=<env>] [--feature=<feature>] [--loglevel=<loglevel>]
  tasks_launcher.py training_autosklearn (--instrument=<instrument>) (--target=<target>) [--env=<env>] [--feature=<feature>] [--loglevel=<loglevel>]
  tasks_launcher.py training_autogluon (--instrument=<instrument>) [--estimator=<estimator>] (--target=<target>) [--env=<env>] [--feature=<feature>] [--loglevel=<loglevel>]
  tasks_launcher.py training_pycaret (--instrument=<instrument>) [--estimator=<estimator>] (--target=<target>) [--env=<env>] [--feature=<feature>] [--loglevel=<loglevel>]
  tasks_launcher.py testing_ml (--instrument=<instrument>) (--target=<target>) [--env=<env>] [--loglevel=<loglevel>]
  tasks_launcher.py (-h | --help)
  tasks_launcher.py --version

Options:
  -e --env=<env>                Environment.
  -m --estimator=<estimator>    Estimator.
  -t --target=<target>          Target.
  -f --feature=<feature>        Feature.
  -s --signaler=<signaler>      Signaler.
  -i --instrument=<instrument>  Instrument.
  -y --start-date=<start-date>  Start date.
  -z --end-date=<end-date>      End date.
  -l --loglevel=<loglevel>      Set log level [default: INFO]
  -d --date=<date>              Date.
  -p --past=<past>              Past.
  -h --help                     Show this screen.
  -v --version                  Show version.
"""
import os
import sys

from docopt import docopt
from logbook import StreamHandler, set_datetime_format, FileHandler, NestedSetup

from varatra_tasks.launchers import backtesting, features_creator, training_tpot, training_automl, testing_ml, backup, \
    training_autogluon, training_autokeras, training_autosklearn, features_importance, training_pycaret


def main(arguments):
    if arguments['backtesting']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()
        backtesting.launch(
            arguments['--signaler'],
            arguments['--instrument'],
            start_date=None,
            end_date=None,
            last_hours=int(arguments['--past']),
            env_context=env_context
        )
    elif arguments['backup']:
        backup.launch(
            arguments['--instrument'],
            start_date=int(arguments['--start-date']),
            end_date=int(arguments['--end-date'])
        )

    elif arguments['features_creator']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()
        # check features
        if arguments['--feature']:
            features = str(arguments['--feature']).split(",")
        else:
            features = []
        features_creator.launch(
            arguments['--instrument'],
            features,
            date=arguments['--date'],
            env_context=env_context
        )

    elif arguments['features_importance']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()
        # check features
        if arguments['--feature']:
            features = str(arguments['--feature']).split(",")
        else:
            features = []
        features_importance.launch(
            arguments['--instrument'],
            features,
            arguments['--target'],
            env_context=env_context
        )

    elif arguments['training_tpot']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()

        # feature list
        if arguments['--feature']:
            features = arguments['--feature'].split(",")
        else:
            features = list()

        # estimator list
        if arguments['--estimator']:
            estimators = arguments['--estimator'].split(",")
        else:
            estimators = list()

        training_tpot.launch(
            arguments['--instrument'],
            arguments['--target'],
            features=features,
            estimators=estimators,
            env_context=env_context
        )

    elif arguments['training_automl']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()

        # feature list
        if arguments['--feature']:
            features = arguments['--feature'].split(",")
        elif arguments['--feature']:
            features = list()

        training_automl.launch(
            arguments['--instrument'],
            arguments['--target'],
            features=features,
            env_context=env_context
        )

    elif arguments['training_autokeras']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()

        # feature list
        if arguments['--feature']:
            features = arguments['--feature'].split(",")
        elif arguments['--feature']:
            features = list()

        training_autokeras.launch(
            arguments['--instrument'],
            arguments['--target'],
            features=features,
            env_context=env_context
        )

    elif arguments['training_autosklearn']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()

        # feature list
        if arguments['--feature']:
            features = arguments['--feature'].split(",")
        elif arguments['--feature']:
            features = list()

        training_autosklearn.launch(
            arguments['--instrument'],
            arguments['--target'],
            features=features,
            env_context=env_context
        )

    elif arguments['training_autogluon']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()

        # feature list
        if arguments['--feature']:
            features = arguments['--feature'].split(",")
        else:
            features = list()

        # estimator list
        if arguments['--estimator']:
            estimators = arguments['--estimator'].split(",")
        else:
            estimators = list()

        training_autogluon.launch(
            arguments['--instrument'],
            arguments['--target'],
            features=features,
            estimators=estimators,
            env_context=env_context
        )

    elif arguments['training_pycaret']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()

        # feature list
        if arguments['--feature']:
            features = arguments['--feature'].split(",")
        else:
            features = list()

        # estimator list
        if arguments['--estimator']:
            estimators = arguments['--estimator'].split(",")
        else:
            estimators = list()

        # training pycaret
        training_pycaret.launch(
            arguments['--instrument'],
            arguments['--target'],
            features=features,
            estimators=estimators,
            env_context=env_context
        )

    elif arguments['testing_ml']:
        if arguments['--env']:
            env_context = dict([x.split("=") for x in arguments['--env'].split(",")])
        else:
            env_context = dict()
        testing_ml.launch(
            arguments['--instrument'],
            arguments['--target'],
            env_context=env_context
        )


if __name__ == '__main__':
    # parse arguments
    arguments = docopt(__doc__, version='Tasks Launcher v1.0')

    set_datetime_format("local")

    # subscribe to instruments
    format_string = (
        '[{record.time:%Y-%m-%d %H:%M:%S.%f%z}] '
        '{record.level_name: <5} - {record.channel: <15}: {record.message}'
    )

    # log file
    log_file = os.path.join(os.path.dirname(__file__), '..', 'varatra_logs', 'varatra_tasks.log')

    # create directory
    logs_dir = os.path.dirname(log_file)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # handlers
    handlers = [
        StreamHandler(sys.stdout, level=arguments['--loglevel'], format_string=format_string, bubble=True),
        FileHandler(log_file, level='INFO', format_string=format_string, bubble=True),
    ]

    # log handlers
    with NestedSetup(handlers):
        main(arguments)
