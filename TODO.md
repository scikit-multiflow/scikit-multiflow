 - create two event observers: one for algorithms and one for drift detection.

#TODO: add
 - strategies on event streaming: shall stop after q or given time window?
 - strategies to compute features on the fly
 - evaluation strategy: time delayed prequential
 - results observer, supporting async results reporting + policies to discard data that arrives too late?
 - add documentation on how to use Docker for this purpose
 
 # DONE
  - data -> generator
  - data -> source
  - data -> stream
  - data -> observer
  - evaluation strategy: holdout vs prequential


How to test:
pytest --capture=tee-sys tests/data/observer/test_evaluation_event_observer.py --showlocals -v


 - the issue we want to get assigned to push changes: https://github.com/scikit-multiflow/scikit-multiflow/issues/106
 - our PRs so far:
   - developer tools: https://github.com/scikit-multiflow/scikit-multiflow/pull/276
