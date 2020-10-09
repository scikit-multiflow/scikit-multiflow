from skmultiflow.data.observer.train_eval_trigger import QuantityBasedHoldoutTrigger


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
        assert train_eval_trigger.shall_fit() == True


    eventcount = 0
    while eventcount < test_size:
        train_eval_trigger.update({})
        assert train_eval_trigger.shall_predict() == True
        assert train_eval_trigger.shall_fit() == False
        eventcount += 1

    train_eval_trigger.update({})
    assert train_eval_trigger.shall_predict() == False
    assert train_eval_trigger.shall_fit() == True
