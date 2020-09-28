from skmultiflow.data.observer.prequential_trigger import PrequentialTrigger


def test_prequential_trigger():
    events_to_wait = 5
    train_eval_trigger = PrequentialTrigger(events_to_wait)

    eventcount = 0
    while (eventcount < events_to_wait):
        train_eval_trigger.update({})
        assert train_eval_trigger.shall_predict() == False
        assert train_eval_trigger.shall_fit() == True
        eventcount += 1

    train_eval_trigger.update({})
    assert train_eval_trigger.shall_predict() == True
    assert train_eval_trigger.shall_fit() == True
