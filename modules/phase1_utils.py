"""
    Nelson Farrell & Michael Massone
    Final Project
    DS 5220 Supervised Machine Learning
    09-22-2024

    This file contains utility functions that can be used when performing a test train/split
"""
################################################################################################################################
# Packages
import pandas as pd
from sklearn.model_selection import train_test_split

################################################################################################################################
# Index

#1  perform_the_train_test_split()
#2 report_check_split_details_save_data_sets()
#3 get_missingness()


################################################################################################################################
# 1

def perform_the_train_test_split(df: pd.DataFrame,
                                 target_attr: str, 
                                 test_size: float, 
                                 train_test_split_random_state: int, 
                                 split_folder: str, 
                                 prefix: str = None, 
                                 val: bool = False, 
                                 stratify: bool = False) -> dict:
    """
    Performs train/test or train/validation split on a dataset, saves the results to disk, and returns the training set.

    Args:
        df (pd.DataFrame): 
            The dataframe to be split into train and test/validation sets. 
        target_attr (str)    
            The name of the target variable column.
        test_size (float): 
            The proportion of the dataset to allocate to the test/validation set (e.g., 0.2 for a 20% test set).
        train_test_split_random_state (int): 
            Random state for reproducibility when performing the split.
        split_folder (str): 
            Directory where the train/test (or train/validation) CSV files will be saved.
        prefix (str, optional): 
            Prefix for the filenames to be saved. If not provided, no prefix will be used.
        val (bool, optional): 
            If True, performs a train/validation split instead of train/test. The smaller split is saved as 'validation_df.csv'.
        stratify (bool, optional): 
            If True, the split will be stratified based on the target variable to ensure class balance in both splits.

    Returns:
        dict: 
            A dictionary containing the training features (X) and the target (y) as pandas DataFrames:
            - 'train_cap_x_df': Training features.
            - 'train_y_df': Training target variable.
    
    The function will also save the following CSV files in the specified `split_folder`:
    - 'train_df.csv' for the training set.
    - 'test_df.csv' or 'validation_df.csv' depending on the `val` argument for the test/validation set.
    """

    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '_'

    if val:
        small_set_name = 'validation_df.csv'
    else:
        small_set_name = 'test_df.csv'

    # Split the dataframe into features (cap_x) and target (y)
    #cap_x_df, y_df = df.iloc[:, :-1], df.iloc[:, -1].to_frame()
    y_df = df.loc[:, target_attr].copy().to_frame()
    cap_x_df = df.drop(columns=[target_attr]).copy()
    
    # Apply stratification if specified
    if stratify:
        stratify = y_df
    else:
        stratify = None

    # Perform the train/test or train/validation split
    train_cap_x_df, test_cap_x_df, train_y_df, test_y_df = train_test_split(
        cap_x_df, y_df, test_size=test_size, random_state=train_test_split_random_state, shuffle=True, stratify=stratify
    )

    # Report, check the split details, and save the datasets
    report_check_split_details_save_data_sets(
        df, train_cap_x_df, train_y_df, small_set_name, split_folder, test_cap_x_df, test_y_df, prefix, stratify
    )

    del test_cap_x_df, test_y_df

    return_dict = {
        'train_cap_x_df': train_cap_x_df,
        'train_y_df': train_y_df
    }

    return return_dict

################################################################################################################################
# 2

def report_check_split_details_save_data_sets(df: pd.DataFrame, 
                                              train_cap_x_df: pd.DataFrame, 
                                              train_y_df: pd.DataFrame, 
                                              small_set_name: str, 
                                              split_folder: str, 
                                              test_cap_x_df: pd.DataFrame, 
                                              test_y_df: pd.DataFrame, 
                                              prefix: str, 
                                              stratify: pd.DataFrame = None):
    """
    Reports the split details, checks the data consistency, and saves the train/test or train/validation datasets to disk.

    Args:
        df (pd.DataFrame): 
            The original dataframe that was split.
        train_cap_x_df (pd.DataFrame): 
            Training set features.
        train_y_df (pd.DataFrame): 
            Training set target variable.
        small_set_name (str): 
            Filename for the smaller dataset (test or validation set).
        split_folder (str): 
            Directory where the datasets will be saved.
        test_cap_x_df (pd.DataFrame): 
            Test/Validation set features.
        test_y_df (pd.DataFrame): 
            Test/Validation set target variable.
        prefix (str): 
            Prefix for the filenames to be saved.
        stratify (pd.DataFrame or None): 
            Stratification target (if any). Used to report class balance.
    """

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

    # Verify index consistency
    assert (list(train_cap_x_df.index) == list(train_y_df.index))
    assert (list(test_cap_x_df.index) == list(test_y_df.index))

    # save the train and test/validation sets to CSV files
    pd.concat([train_cap_x_df, train_y_df], axis=1).to_csv(prefix + split_folder + 'train_df.csv', index=True,
                                                           index_label='index')
    pd.concat([test_cap_x_df, test_y_df], axis=1).to_csv(prefix + split_folder + small_set_name, index=True, index_label='index')

################################################################################################################################
# 3

def get_missingness(df: pd.DataFrame, 
                    missingness_threshold: float) -> dict:
    """
    Identifies and reports the columns in the dataframe that exceed a certain missingness threshold.

    Args:
        df (pd.DataFrame): 
            The dataframe to check for missing values.
        missingness_threshold (float): 
            The threshold proportion of missing values to flag a column for removal (e.g., 0.2 for 20%).

    Returns:
        dict: 
            A dictionary containing the list of columns to be dropped due to high missingness:
            - 'missingness_drop_list': List of column names with missingness exceeding the threshold.
    """

    missingness_drop_list = []
    # ITer cols and check calculate missingness
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

################################################################################################################################
# END
if __name__ == "__main__":
    pass