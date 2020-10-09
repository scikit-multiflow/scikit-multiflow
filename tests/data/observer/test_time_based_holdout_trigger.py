from skmultiflow.data.observer.train_eval_trigger import TimeBasedHoldoutTrigger
from datetime import datetime, timedelta


def get_event_time(x):
    return x['event_time']


def create_event(eventcount, reference_date):
    return {'event_time': reference_date + timedelta(days=eventcount+1)}


def test_time_based_holdout_trigger():
    reference_date = datetime(2012, 3, 5)
    initial_time_window_days = 7
    test_time_window_days = 2
    initial_time_window = timedelta(days=initial_time_window_days)
    wait_to_test_time_window = timedelta(days=5)
    test_time_window = timedelta(days=test_time_window_days)
    train_eval_trigger = TimeBasedHoldoutTrigger(initial_time_window, wait_to_test_time_window, test_time_window, get_event_time)

    eventcount = 0
    while eventcount <= initial_time_window_days:
        train_eval_trigger.update(create_event(eventcount, reference_date))
        assert train_eval_trigger.shall_predict() == False
        assert train_eval_trigger.shall_fit() == True
        eventcount += 1


    test_eventcount = 0
    while test_eventcount <= test_time_window_days:
        train_eval_trigger.update(create_event(eventcount+test_eventcount, reference_date))
        assert train_eval_trigger.shall_predict() == True
        assert train_eval_trigger.shall_fit() == False
        test_eventcount += 1

    train_eval_trigger.update(create_event(eventcount+test_eventcount, reference_date))
    assert train_eval_trigger.shall_predict() == False
    assert train_eval_trigger.shall_fit() == True
