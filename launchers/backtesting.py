import os
import sys

from logbook import StreamHandler

from varatra_patterns.recursionlimit import recursionlimit
from varatra_signaler.core.signaler_backtesting import SignalerBacktesting
from varatra_tasks.core.tasks.backtesting import upload_report, upload_context

StreamHandler(sys.stdout, level=os.getenv('LOG_LEVEL', 'INFO')).push_application()


def launch(signaler_name, instrument, start_date=None, end_date=None, last_hours=0, env_context=None):
    """
    Run backtesting

    """
    output_directory = os.path.join(os.path.expanduser("~"), 'varatra_output')
    backtester = SignalerBacktesting.create_from_name(
        signaler_name,
        [instrument],
        output_directory,
        monitoring=False,
    )

    # update env context
    if env_context:
        os.environ.update(env_context)

    # run backtester
    with recursionlimit(150000):
        df = backtester.start(
            start_date,
            end_date,
            last_hours=last_hours,
            from_db=True,
        )

        # upload env context
        if env_context:
            run_id = backtester.get_signaler_executor().get_signaler().id
            t_upload_context = upload_context.Task("upload_context")
            t_upload_context.run(run_id, env_context)

        # upload report
        t_upload_report = upload_report.Task("upload_report")
        t_upload_report.run(df)


if __name__ == "__main__":
    launch("bbands.percentile", "BITSTAMP_XRP_USD", "2020-01-02", "2020-01-02", last_hours=24)
