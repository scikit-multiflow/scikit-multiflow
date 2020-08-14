import pandas as pd
from skmultiflow.data.ndata_stream import NDataStream
from skmultiflow.trees.nhoeffding_tree_regressor import NRegressionHoeffdingTree
from skmultiflow.evaluation.evaluate_tprequential import EvaluateTPrequential

# 0. Load dataframe from .csv and preprocess to get timestamps, features and target
dataset_filename = "/Users/jmrozanec/other/repo/qlector/perun-sap-etl/data/output/dataset/BASE-DATASET-HORIZON_6W-by-dmkey.csv"
df = pd.read_csv(dataset_filename)

df = df.drop(['dmkey', 'material_id', 'horizon'], axis=1)
remaining_cols = set(df.columns.values)
features = list(remaining_cols.difference(set(['year_month', 'target'])))
df = df.sort_values(by=['year_month'], ascending=True)
t = df[['year_month']]
X = df[list(features)]
y = df[['target']]

# 1. Create a stream
stream = NDataStream(t=t, X=X, y=y)
stream.prepare_for_use()

# 2. Instantiate the HoeffdingTree classifier
ht = NRegressionHoeffdingTree()

# 3. Setup the evaluator
evaluator = EvaluateTPrequential(max_timestamps=24, pretrain_timestamps=3, metrics=['mean_absolute_error'], output_file='results.csv')

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ht)
