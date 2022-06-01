import digdag

from core.prometheus import get_metrics


def run(pattern):
    task = get_metrics.Task("get_metrics")
    instrument_list = task.run(pattern)
    digdag.env.store({"instrument_list": instrument_list})
