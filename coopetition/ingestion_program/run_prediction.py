import sys, os
import yaml

import numpy as np
import pandas as pd

ingestion_program_path, submission_program_path, ref_data_path, output_path = sys.argv[1:]

config_fname = os.path.join(ingestion_program_path, 'config.yml')

with open(config_fname) as f:
    config = yaml.load(f)

input_fname_train = os.path.join(ref_data_path, config['training_file'])
input_fname_test  = os.path.join(ref_data_path, config['scoring_file' ])
output_fname_test_pred = os.path.join(output_path, config['prediction_file'])

df_train = pd.read_csv(input_fname_train)
X_train = df_train[config['X_cols']]
Y_train = df_train[config['Y_cols']]

X_test = pd.read_csv(input_fname_test, usecols=config['X_cols'])

sys.path.append(submission_program_path)
import submission_model

model = submission_model.Model()

model.fit(X_train, Y_train)
Y_test_pred = model.predict(X_test)

assert len(Y_test_pred) == len(X_test)
assert set(Y_test_pred.columns) == set(config['Y_cols'])

Y_test_pred.to_csv(output_fname_test_pred, index=False)
