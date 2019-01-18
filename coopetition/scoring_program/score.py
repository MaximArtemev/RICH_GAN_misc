import sys, os
import yaml

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


scoring_program_path, input_dir, output_dir = sys.argv[1:]

config_file_name = os.path.join(scoring_program_path, 'config.yml')

with open(config_file_name) as f:
    config = yaml.load(f)

submit_dir = os.path.join(input_dir, 'res')
reference_dir = os.path.join(input_dir, 'ref')

reference_file = os.path.join(reference_dir, config['scoring_file'])
prediction_file = os.path.join(submit_dir, config['prediction_file'])

if not os.path.isdir(submit_dir):
    print("{} doesn't exist".format(submit_dir))
elif not os.path.isdir(reference_dir):
    print("{} doesn't exist".format(reference_dir))
else:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reference  = pd.read_csv(reference_file)
    prediction = pd.read_csv(prediction_file)
    # prediction[config['X_cols']] = reference[config['X_cols']]

    score = 0
    cols = config['Y_cols'] + config['X_cols']
    w_normal = np.random.normal(size=(config['n_slices'], len(cols)))
    reference = reference[cols].values
    prediction = prediction[cols].values
    for k in range(config['n_slices']):
        score = max(score,
                    ks_2samp(
                        np.sum(w_normal[k] * reference, axis=1), 
                        np.sum(w_normal[k] * prediction, axis=1)
                    )[0]
                   )

    output_filename = os.path.join(output_dir, 'scores.txt')

    with open(output_filename, 'w') as output_file:
        output_file.write("KS: {}".format(score))
