from pathlib import Path

import click
import pandas as pd

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('data')


def get_data(dir, file):
    df = pd.read_csv(dir.joinpath(file), dtype="str", sep=",", encoding="utf-8", skipinitialspace=True)
    return df


def get_gt(df_test):
    df_clicks = df_test[df_test["action_type"] == "clickout item"]
    df_gt = df_clicks[df_clicks["reference"].notnull()]
    return df_gt


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--out-path', default=None, help='Output directory of preprocessed CSV files')
def main(data_path, out_path):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory

    print('reading test..')
    df_test = get_data(data_directory, 'test.csv')

    print('get groundtruth')
    df_gt = get_gt(df_test)

    df_gt.to_csv(data_directory.joinpath('ground_truth.csv'), sep=',', index=None, header=True)

    print('finished')


if __name__ == '__main__':
    main()
