import os
import sys

from datetime import datetime
from logbook import StreamHandler
from sklearn.model_selection import train_test_split

from varatra_features.core.feature_executor import FeatureExecutor
from varatra_features.core.feature_loader import FeatureLoader
from varatra_tasks.core.tasks.features import load_parquet
from varatra_tasks.core.tasks.features import sync_s3
from varatra_tasks.core.tasks.model import classification_report, confusion_matrix
from varatra_tasks.core.tasks.model import dump_to_s3
from logbook import Logger

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()


def launch(instrument_id, target, env_context=None, features=None):
    """
    Run machine learner

    """
    from auto_ml import Predictor
    from auto_ml.utils_models import load_ml_model
    log = Logger("training_automl")

    if env_context:
        os.environ.update(env_context)

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
    df.dropna(inplace=True)
    log.info(f"msg='dataset' length='{len(df)}'")

    # create X, y
    features_valid = [c for c in df.columns if c in features]

    # run id
    tmp_run_dir = os.path.join("/tmp", datetime.now().strftime('%Y%M%d_%H%M%S'))
    os.makedirs(tmp_run_dir)
    model_file = os.path.join(tmp_run_dir, f"model.joblib")

    # output
    df_train, df_test = train_test_split(df, train_size=0.75, test_size=0.25)
    column_descriptions = {
        target: 'output'
    }
    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_train)

    # Score the model on test data
    X_test = df_test[features_valid]
    y_test = df_test[target]
    test_modello = ml_predictor.save(model_file)
    trained_model = load_ml_model(test_modello)

    # classification report
    t_cls_report = classification_report.Task("classification_report")
    try:
        clf_report = t_cls_report.run(trained_model, X_test, y_test)
        clf_report_dict = clf_report.dict()
    except Exception as error:
        log.error(f"msg='failed to generate the classification report' error='{error}'")
        clf_report_dict = dict()

    # confusion matrix
    t_confusion_matrix = confusion_matrix.Task("confusion_matrix")
    try:
        confusion_matrix_report = t_confusion_matrix.run(trained_model, X_test, y_test)
        confusion_matrix_dict = confusion_matrix_report.to_dict()
    except Exception as error:
        log.error(f"msg='failed to generate the confusion matrix' error='{error}'")
        confusion_matrix_dict = dict()

    score = ml_predictor.score(X_test, y_test)

    meta = {
        'features': features_valid,
        'target': target,
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1]),
        'size': len(df),
        'training': {
            'start_date': str(df_train.index[0]),
            'end_date': str(df_train.index[-1]),
            'size': len(df_train),
        },
        'testing': {
            'start_date': str(df_test.index[0]),
            'end_date': str(df_test.index[-1]),
            'size': len(df_test),
        },
        'classification_report':clf_report_dict,
        'confusion_matrix': confusion_matrix_dict,
        'score': score
    }
    # dump to s3
    t_dump_to_s3 = dump_to_s3.Task("dump_to_s3")
    t_dump_to_s3.run(instrument_id, features_valid, target, 'AUTOML', 'CLASSIFIER', model_file, meta)


if __name__ == "__main__":
    launch("OANDA_EUR_USD", 'profit__binary__10M')
