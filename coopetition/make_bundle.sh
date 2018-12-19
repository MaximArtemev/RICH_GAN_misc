# one-liner taken from https://stackoverflow.com/a/246128/3801744
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

zip -j $DIR/bundle/data_train.zip $DIR/../data_csv/v1_*_train.csv
zip -j $DIR/bundle/data_validation.zip $DIR/../data_csv/v1_*_validation.csv
zip -j $DIR/bundle/data_test.zip $DIR/../data_csv/v1_*_test.csv

zip -j $DIR/bundle/example_solution.zip $DIR/example_solution/*
zip -j $DIR/bundle/ingestion_program.zip $DIR/ingestion_program/*
zip -j $DIR/bundle/scoring_program.zip $DIR/scoring_program/*
