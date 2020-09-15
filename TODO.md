 - create two event observers: one for algorithms and one for drift detection.

#TODO: add
 - strategies to compute features on the fly
 - evaluation strategy: holdout vs prequential
 - results observer, supporting async results reporting + policies to discard data that arrives too late?
 - add documentation on how to use Docker for this purpose
 
 # DONE
  - data -> generator
  - data -> source
  - data -> stream
  - data -> observer


How to test:
pytest --capture=tee-sys tests/data/observer/test_evaluation_event_observer.py --showlocals -v
