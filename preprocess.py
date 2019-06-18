from pathlib import Path

import click
import pandas as pd
import random
import os

random.seed(100)

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('data')


def get_data(dir, file):
    df = pd.read_csv(dir.joinpath(file), dtype="str", sep=",", encoding="utf-8", skipinitialspace=True)
    return df


def get_users_to_keep(df):
    all_users = df.user_id.unique()
    keep_users = random.sample(list(all_users), round(all_users.size * 0.1))  # keep 10% of the users
    return keep_users


def subsample(df, keepusers):
    df_subsample = df[df.user_id.isin(keepusers)]
    return df_subsample


def get_gt(path):
    df_test = pd.read_csv(path)
    mask_click_out = df_test["action_type"] == "clickout item"
    df_clicks = df_test[mask_click_out]

    mask_ground_truth = df_clicks["reference"].notnull()
    df_gt = df_clicks[mask_ground_truth]
    return df_gt


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--out-path', default=None, help='Output directory of preprocessed CSV files')
def main(data_path, out_path):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    output_directory = data_directory.joinpath('preprocess')

    print('reading train..')
    df_non_test = get_data(data_directory, 'train.csv')

    print('reading test..')
    df_test = get_data(data_directory, 'test.csv')

    print('get keepable users list')
    keepable_users = get_users_to_keep(df_non_test)

    print('start filtering')
    df_non_test_sub = subsample(df_non_test, keepable_users)
    df_ds_test_sub = subsample(df_test, keepable_users)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    df_non_test_sub.to_csv(output_directory.joinpath('test_subsample.csv'), sep=',', index=None, header=True)
    df_ds_test_sub.to_csv(output_directory.joinpath('train_subsample.csv'), sep=',', index=None, header=True)

    print('finished')


if __name__ == '__main__':
    main()
