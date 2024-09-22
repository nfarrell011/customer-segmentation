"""
    Nelson Farrell & Michael Massone
    Final Project
    DS 5220 Supervised Machine Learning
    09-22-2024

    This file contains utility functions that can be used when performing a test train/split
"""
# Packages
import pandas as pd
from sklearn.model_selection import train_test_split


def perform_the_train_test_split(df:pd.DataFrame, test_size:float, train_test_split_random_state:int, split_folder:str, prefix:str=None, val:bool=False, stratify:bool=False) -> dict:
    """
    Performs train/test split on a dataset of interest.

    Args:
     * df: (pd.Dataframe)                       - The dataframe to be split into train and test sets.
     * test_size: (float)                       - The proportion of the data to hold in the test set.
     * prefix: (str)                            -

     if val = False then train/test split
    # if val = True then train/validation split - only difference is that the test set is saved as the val set
    
    """
    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '_'

    if val:
        small_set_name = 'validation_df.csv'
    else:
        small_set_name = 'test_df.csv'

    cap_x_df, y_df = df.iloc[:, :-1], df.iloc[:, -1].to_frame()
    if stratify:
        stratify = y_df
    else:
        stratify = None

    train_cap_x_df, test_cap_x_df, train_y_df, test_y_df = \
        train_test_split(cap_x_df, y_df, test_size=test_size, random_state=train_test_split_random_state, shuffle=True,
                         stratify=stratify)

    report_check_split_details_save_data_sets(df, train_cap_x_df, train_y_df, small_set_name, split_folder, test_cap_x_df, test_y_df,
                                              prefix, stratify)

    del test_cap_x_df, test_y_df

    return_dict = {
        'train_cap_x_df': train_cap_x_df,
        'train_y_df': train_y_df
    }

    return return_dict


def report_check_split_details_save_data_sets(df, train_cap_x_df, train_y_df, small_set_name, split_folder, test_cap_x_df, test_y_df,
                                              prefix, stratify):
    print(25 * '*')
    print('\ndf.shape:')
    print(df.shape)
    target_attr = None
    if stratify is not None:
        target_attr = train_y_df.columns[0]
        print(f'\ntarget class fractional balance:\n{df[target_attr].value_counts()/df.shape[0]}', sep='')

    print('\n', 25 * '*', sep='')
    print('\ntrain_df.csv:')
    print(train_cap_x_df.shape, train_y_df.shape)
    if stratify is not None:
        print(f'\ntarget class fractional balance:\n{train_y_df[target_attr].value_counts()/train_y_df.shape[0]}',
              sep='')

    print('\n', 25 * '*', sep='')
    print('\n', small_set_name, sep='')
    print(test_cap_x_df.shape, test_y_df.shape)
    if stratify is not None:
        print(f'\ntarget class fractional balance:\n{test_y_df[target_attr].value_counts()/test_y_df.shape[0]}',
              sep='')

    assert (list(train_cap_x_df.index) == list(train_y_df.index))
    assert (list(test_cap_x_df.index) == list(test_y_df.index))

    pd.concat([train_cap_x_df, train_y_df], axis=1).to_csv(prefix + split_folder + 'train_df.csv', index=True,
                                                           index_label='index')
    pd.concat([test_cap_x_df, test_y_df], axis=1).to_csv(prefix + split_folder + small_set_name, index=True, index_label='index')


def get_missingness(df, missingness_threshold):

    missingness_drop_list = []
    for attr in df.columns:
        attr_missingness = df[attr].isna().sum() / df.shape[0]
        print(f'{attr} missingness = {attr_missingness}')
        if attr_missingness >= missingness_threshold:
            missingness_drop_list.append(attr)

    print(f'\nmissingness_drop_list:\n{missingness_drop_list}')

    return_dict = {
        'missingness_drop_list': missingness_drop_list
    }

    return return_dict


if __name__ == "__main__":
    pass