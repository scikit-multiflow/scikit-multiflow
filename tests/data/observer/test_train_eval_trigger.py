from skmultiflow.data.observer.train_eval_trigger import QuantityBasedHoldoutTrigger, TimeBasedHoldoutTrigger
from skmultiflow.data.observer.train_eval_trigger import PrequentialTrigger, TimeBasedCrossvalidationTrigger
from datetime import datetime, timedelta


def get_event_time(x):
    return x['event_time']


def create_event(eventcount, reference_date):
    return {'event_time': reference_date + timedelta(days=eventcount+1)}


def test_prequential_trigger():
    events_to_wait = 5
    train_eval_trigger = PrequentialTrigger(events_to_wait)

    eventcount = 0
    while (eventcount < events_to_wait):
        train_eval_trigger.update({})
        assert train_eval_trigger.shall_predict() == False
        assert train_eval_trigger.shall_buffer() == True
        eventcount += 1

    train_eval_trigger.update({})
    assert train_eval_trigger.shall_predict() == True
    assert train_eval_trigger.shall_buffer() == True


def test_quantity_based_holdout_trigger_first_time_wait_less_than_wait_to_test():
    first_time_wait = 5
    n_wait_to_test = 10
    test_size = 2
    train_eval_trigger = QuantityBasedHoldoutTrigger(first_time_wait, n_wait_to_test, test_size)

    eventcount = 0
    while eventcount < max(first_time_wait, n_wait_to_test):
        train_eval_trigger.update({})
        eventcount += 1
        assert train_eval_trigger.shall_predict() == False
        assert train_eval_trigger.shall_buffer() == True


    eventcount = 0
    while eventcount < test_size:
        train_eval_trigger.update({})
        assert train_eval_trigger.shall_predict() == True
        assert train_eval_trigger.shall_buffer() == False
        eventcount += 1

    train_eval_trigger.update({})
    assert train_eval_trigger.shall_predict() == False
    assert train_eval_trigger.shall_buffer() == True


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
        assert train_eval_trigger.shall_buffer() == True
        eventcount += 1


    test_eventcount = 0
    while test_eventcount <= test_time_window_days:
        train_eval_trigger.update(create_event(eventcount+test_eventcount, reference_date))
        assert train_eval_trigger.shall_predict() == True
        assert train_eval_trigger.shall_buffer() == False
        test_eventcount += 1

    train_eval_trigger.update(create_event(eventcount+test_eventcount, reference_date))
    assert train_eval_trigger.shall_predict() == False
    assert train_eval_trigger.shall_buffer() == True


def test_time_crossvalidation_trigger():
    reference_date = datetime(2012, 3, 5)
    initial_time_window_days = 7
    test_time_window_days = 2
    initial_time_window = timedelta(days=initial_time_window_days)
    wait_to_test_time_window = timedelta(days=5)
    test_time_window = timedelta(days=test_time_window_days)
    train_eval_trigger = TimeBasedCrossvalidationTrigger(initial_time_window, wait_to_test_time_window, test_time_window, get_event_time)

    eventcount = 0
    while eventcount <= initial_time_window_days:
        train_eval_trigger.update(create_event(eventcount, reference_date))
        assert train_eval_trigger.shall_predict() == False
        assert train_eval_trigger.shall_buffer() == True
        eventcount += 1

    test_eventcount = 0
    while test_eventcount <= test_time_window_days:
        train_eval_trigger.update(create_event(eventcount+test_eventcount, reference_date))
        assert train_eval_trigger.shall_predict() == True
        assert train_eval_trigger.shall_buffer() == True
        test_eventcount += 1

    train_eval_trigger.update(create_event(eventcount+test_eventcount, reference_date))
    assert train_eval_trigger.shall_predict() == True
    assert train_eval_trigger.shall_buffer() == True
