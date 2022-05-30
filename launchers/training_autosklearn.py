import os
import sys

from joblib import dump
from logbook import StreamHandler
from sklearn.model_selection import train_test_split

from varatra_features.core.feature_executor import FeatureExecutor
from varatra_features.core.feature_loader import FeatureLoader
from varatra_tasks.core.tasks.features import load_parquet
from varatra_tasks.core.tasks.features import sync_s3
from varatra_tasks.core.tasks.model import classification_report, confusion_matrix
from varatra_tasks.core.tasks.model import dump_to_s3

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()
from datetime import datetime
from logbook import Logger
import time
from tabulate import tabulate


def launch(instrument_id, target, env_context=None, features=None, shuffle=False):
    """
    Run machine learner

    """
    from autosklearn import metrics
    import autosklearn.classification

    start_time = time.time()

    log = Logger("training_autosklearn")
    if env_context:
        os.environ.update(env_context)

    # shuffle
    shuffle = os.getenv("SHUFFLE", shuffle)

    # select features
    features_dir = FeatureExecutor.get_features_dir()
    features = FeatureLoader.filter_features(features_dir, features)
    features_and_target = features + [target]

    # sync parquet
    t_sync_s3 = sync_s3.Task("sync_s3")
    exchange, symbol = instrument_id.split("_", 1)
    parquet_filter = {'exchange': exchange, 'symbol': symbol, 'features': features_and_target}
    t_sync_s3.run(parquet_filter=parquet_filter)

    # drop nan values from dataframe
    t_load_parquet = load_parquet.Task("load_parquet")
    df = t_load_parquet.run(instrument_id, feature_list=features_and_target)

    # clean data
    log.info(f"msg='dataset length after loading' length='{len(df)}'")
    len_after_load = len(df)
    # print nan
    df_count = df.count().sort_values(ascending=False).to_frame()
    print(tabulate(df_count, headers=['count']))
    # drop nan
    df.dropna(inplace=True)
    log.info(f"msg='dataset length after dropping nan' length='{len(df)}' dropped='{len_after_load - len(df)}'")

    # create X, y
    features_valid = [c for c in df.columns if c in features]
    X = df[features_valid]
    y = df[target]

    # create train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42,
                                                        shuffle=shuffle)
    log.info(
        f"msg='dataset splitted to train/test' training_length='{len(X_train)}' testing_length='{len(X_test)}' shuffle='{shuffle}''")

    max_memory = int(os.getenv("MAX_MEMORY", 0))
    n_jobs = int(os.getenv("N_JOBS", -1))

    if max_memory:
        log.info(f"msg='training started' memory='{max_memory}'")
        automl = autosklearn.classification.AutoSklearnClassifier(
            ml_memory_limit=max_memory,
            time_left_for_this_task=10800,
            tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
            output_folder='/tmp/autosklearn_parallel_1_example_out',
            n_jobs=n_jobs,
            seed=5,
            delete_output_folder_after_terminate=False,
            delete_tmp_folder_after_terminate=False,
        )
    else:
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=10800)

    automl.fit(X_train, y_train, metric=metrics.precision)
    # end time
    end_time = time.time()

    # run id
    tmp_run_dir = f"/tmp/{datetime.now().strftime('%Y%M%d_%H%M%S')}"
    os.makedirs(tmp_run_dir)
    model_file = os.path.join(tmp_run_dir, f"model.joblib")

    # dump output
    log.info("msg='dumping model to file'")
    dump(automl, model_file)

    # classification report
    t_cls_report = classification_report.Task("classification_report")
    clf_report = t_cls_report.run(automl, X_test, y_test)

    # classification report
    t_confusion_matrix = confusion_matrix.Task("confusion_matrix")
    confusion_matrix_report = t_confusion_matrix.run(automl, X_test, y_test)

    meta = {
        'features': features_valid,
        'target': target,
        'duration': end_time - start_time,
        'dataset': {
            'start_date': str(X.index[0]),
            'end_date': str(X.index[-1]),
            'size': len(X),
            'shuffle': shuffle,
            'training': {
                'start_date': str(X_train.index[0]),
                'end_date': str(X_train.index[-1]),
                'size': len(X_train),
            },
            'testing': {
                'start_date': str(X_test.index[0]),
                'end_date': str(X_test.index[-1]),
                'size': len(X_test),
            }
        },
        'classification_report': clf_report.to_dict(),
        'confusion_matrix': confusion_matrix_report.to_dict()
    }

    # dump to s3
    t_dump_to_s3 = dump_to_s3.Task("dump_to_s3")
    t_dump_to_s3.run(instrument_id, features_valid, target, 'AUTOSKLEARN', 'CLASSIFIER', model_file, meta)


if __name__ == "__main__":
    launch("BINANCE_BTC_USDT", 'profit__binary__10M')
