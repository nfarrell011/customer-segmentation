"""
    Nelson Farrell & Michael Massone
    Final Project - Phase 2 EDA
    DS 5220 Supervised Machine Learning
    10-15-2024

    This file contains utility functions that for EDA.
"""
################################################################################################################################
# Packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

################################################################################################################################
# Index

#  1 -> separate_unique_columns()
#  2 -> check_column_types_and_nans()
#  3 -> plot_bar()
#  4 -> check_frequency()
#  5 -> nominal_eda()
#  6 -> is_discrete()
#  7 -> plot_column()
#  8 -> plot_boxplot()
#  9 -> detect_outliers_zscore()
# 10 -> detect_outliers_iqr()
# 11 -> numerical_eda()


################################################################################################################################
# 1

def separate_unique_columns(df) -> dict:
    """
    This function separates columns of a DataFrame into those with 100% unique values and those with less than 100% unique values.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    dict: A dictionary with two keys: '100% unique' and '<100% unique'. Each contains a list of column names.
    """
    unique_cols = []
    non_unique_cols = []

    for column in df.columns:
        num_unique = df[column].nunique()
        total_values = df[column].size
        
        if num_unique == total_values:
            unique_cols.append(column)
        else:
            non_unique_cols.append(column)
    
    print("*****************************")
    print("non_ML_attr")
    for attr in unique_cols:
        print(attr)
    print("*****************************")
    print("ML_attr")
    for attr in non_unique_cols:
        print(attr)

    return {
        'non_ML_attr': unique_cols,
        'ML_attr': non_unique_cols
    }

################################################################################################################################
# 2

def check_column_types_and_nans(df, threshold=0.2) -> dict:
    '''
    Checks each column of a dataframe and returns a list of numerical attributes and nominal attributes, 
    along with NaN counts for each column
    
    Inputs:
        df(pd.DataFrame): Dataframe to be checked.
        threhsold(float): NAN ratio threshold to add cols to exclusion list. Default is 0.20.
    Out:
        dict{
            'numerical_cols': numerical_cols,     
            'nominal_cols': non_numerical_cols,       
            }
    '''
    
    numerical_cols = []
    num_nan_count = []
    non_numerical_cols = []
    nom_nan_count = []
    nan_above_thresold = []
    
    # Iterate through each column in the DataFrame
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
            nan_count = df[col].isna().sum()
            num_nan_count.append(nan_count) # Count NaN values for the column
            nan_above_thresold.append(col) if nan_count/df.shape[0] > threshold else None
        else:
            non_numerical_cols.append(col)
            nan_count = df[col].isna().sum()
            nom_nan_count.append(nan_count)  
            nan_above_thresold.append(col) if nan_count/df.shape[0] > threshold else None

    
    # Print out the results with Dtype and NaN counts
    print("*"*50)
    print("Numerical Columns:")
    if not numerical_cols:
        print("**NONE**")
    else:    
        for col, nan_count in zip(numerical_cols, num_nan_count):
            print(f"{col} \n - dtype: {df[col].dtype} \n - NaN count: {nan_count} \n - NaN ratio: {nan_count/df.shape[0]}")
    
    print("*"*50)    
    print("Non-Numerical Columns:")
    if not non_numerical_cols:
        print("**NONE**")
    else:
        for col, nan_count in zip(non_numerical_cols, nom_nan_count):
            print(f"{col} \n - dtype: {df[col].dtype} \n - NaN count: {nan_count} \n - NaN ratio: {nan_count/df.shape[0]}")

    print("*"*50)
    print(f"Columns with NAN ration greather than {threshold * 100}%:")
    if not nan_above_thresold:
        print("**NONE**")
    else:
        for col in nan_above_thresold:
            print(col)
            
    return {'numerical_cols': numerical_cols, 'nominal_cols': non_numerical_cols, 'nan_above_threshold': nan_above_thresold}

################################################################################################################################
# 3

def plot_bar(df: pd.DataFrame, column: str):
        """
        Generate a bar plot showing the distribution of values for a given column.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame.
        column : str
            The column name to plot.
        
        Returns:
        --------
        None
        """
        try:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=column, data=df, hue=column, legend=False, palette="muted")            
            plt.title(f'Distribution of {column}', fontsize=14)
            plt.xlabel(column, fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.ylabel('Count', fontsize=12)
            plt.show()
        except Exception as e:
            print(f"Error generating plot for {column}: {e}")

################################################################################################################################
# 4

def check_frequency(df: pd.DataFrame, column: str):
    """
    Calculate and display the frequency distribution of a given column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The column name to analyze.

    Returns:
    --------
    None
    """
    try:
        frequency = df[column].value_counts()
        frequency_ratios = round(df[column].value_counts(normalize=True) * 100, 1)
        freq_df = pd.DataFrame({
            'Count': frequency,
            'Frequency_Ratio (%)': frequency_ratios
        })
        print(f"\nFrequency Distribution for {column}:\n{'_'*50}")
        print(freq_df, "\n")
    except Exception as e:
        print(f"Error calculating frequency for {column}: {e}")

################################################################################################################################

def generate_contingency_tables(df: pd.DataFrame, column: str, target: str):
    """
    Generate contingency tables for a categorical variables in the DataFrame 
    against the specified target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the column to generate contingency table for. 
    target : str
        The name of the target column.

    Returns:
    --------
    None - Prints contingency table.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    print(f"\nContingency Table: {column} vs {target}")
    contingency_table = pd.crosstab(df[column], df[target])

    # Convert counts to percentages of the total
    total = contingency_table.values.sum()
    contingency_table_percentage = (contingency_table / total) * 100

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table_percentage, annot=True, cmap='RdBu', fmt='.2f', linewidths=0.5, cbar=True, vmax=100, vmin=0, center=50)
    plt.title(f'Heatmap: {column} vs {target} (%)', fontsize=16)
    plt.xlabel(target, fontsize=12)
    plt.ylabel(column, fontsize=12)

    # Show the plot
    plt.show()




################################################################################################################################
# 5

def nominal_eda(df: pd.DataFrame, target: str = None):
    """
    Perform Exploratory Data Analysis (EDA) on nominal (categorical) columns in a DataFrame.
    This function generates bar plots and frequency distributions for each nominal column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing nominal columns for analysis.
    target : str, optional
        The name of the target column, if one exists. If provided, it will be highlighted
        in the output to distinguish it from other nominal columns.

    Returns:
    --------
    None
        Displays plots and prints frequency distributions for each nominal column.
    """

    # Iterate through each column and perform EDA
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'object':
            if target and col == target:
                print("*"*27, "\n****** TARGET COLUMN ******\n", "*"*27)
            else:
                generate_contingency_tables(df, col, target)
            plot_bar(df, col)
            check_frequency(df, col)
        else:
            print(f"Skipping non-categorical column: {col}")



################################################################################################################################
# 6

def is_discrete(df: pd.DataFrame, column: str,  threshold: float = 0.05) -> bool:
    """
    Detect if a column is discrete based on its unique values.

    Parameters:
    -----------
    data : pd.Series
        The column to analyze.
    threshold : float, optional (default=0.05)
        The ratio of unique values to total values below which the data is treated as discrete.

    Returns:
    --------
    bool : True if the column is likely discrete, otherwise False.
    """
    unique_ratio = df[column].nunique() / len(df[column])
    all_integers = pd.api.types.is_integer_dtype(df[column])

    # Treat as discrete if the unique ratio is small or if all values are integers
    return unique_ratio < threshold or all_integers
################################################################################################################################
# 7

def plot_column(df: pd.DataFrame, column: str, bins: int = 20, bw_adjust: float = 0.5):
    """
    Plot a column as either a discrete or continuous distribution based on its characteristics.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the column to plot.
    bins : int, optional (default=20)
        Number of bins for continuous histograms.
    bw_adjust : float, optional (default=0.5)
        Bandwidth adjustment for KDE in continuous data.

    Returns:
    --------
    None - Displays the appropriate plot for the column.
    """
    try:
        data = df[column].dropna()

        if is_discrete(df, column):
            print(f"Plotting '{column}' as discrete data.")
            plt.figure(figsize=(8, 5))
            sns.histplot(data, bins=data.nunique(), discrete=True, edgecolor='black')
            sns.rugplot(data, height=0.05, color='black')
        else:
            print(f"Plotting '{column}' as continuous data.")
            plt.figure(figsize=(8, 5))
            sns.histplot(data, bins=bins, kde=True, edgecolor='black')

        plt.title(f'Distribution of {column}', fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.show()

    except Exception as e:
        print(f"Error plotting column '{column}': {e}")

################################################################################################################################
# 8

def plot_boxplot(df: pd.DataFrame, column: str):
    """
    Plot a boxplot to visualize the spread and detect potential outliers using Seaborn.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the numerical column to plot.

    Returns:
    --------
    None - Displays the boxplot.
    """
    try:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[column].dropna())
        plt.title(f'Boxplot of {column}', fontsize=14)
        plt.show()
    except Exception as e:
        print(f"Error generating boxplot for {column}: {e}")

################################################################################################################################
# 9

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """
    Detect outliers in a numerical column using the Z-score method.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the numerical column to analyze.
    threshold : float, optional (default=3.0)
        The Z-score threshold for identifying outliers.

    Returns:
    --------
    Dict : Dictionary containing outliers indices by column key.

    """
    try:
        z_scores = zscore(df[column], nan_policy='omit')
        outliers = df[column][(z_scores > threshold) | (z_scores < -threshold)]
        if not outliers.empty:
            print(f'\nNumber of Outliers in {column} using Zscore with Threshold of {threshold} stds: {len(outliers)}')
            print(outliers)
        else:
            print(f'No significant outliers detected in {column} using Zscore with Threshold of {threshold} stds.')

        return {column: outliers.index}
        
    except Exception as e:
        print(f"Error detecting outliers for {column}: {e}")

################################################################################################################################
# 10

def detect_outliers_iqr(df: pd.DataFrame, column: str):
    """
    Detect outliers in a numerical column using the IQR method.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the numerical column to analyze.

    Returns:
    --------
    Dict : Dictionary containing outliers indices by column key.

    """
    try:
        # Drop missing values for the selected column
        data = df[column].dropna()

        # Compute the IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        # Define the outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        if not outliers.empty:
            print(f"Number of Outliers detected in '{column}' using IQR: {len(outliers)}")
            print(outliers)
        else:
            print(f'No significant outliers detected in {column} using IQR.')

        return {column: outliers.index}

    except Exception as e:
        print(f"Error detecting outliers for column '{column}': {e}")

################################################################################################################################
# 11

def numerical_eda(df: pd.DataFrame, target: str = None):
    """
    Perform Exploratory Data Analysis (EDA) on numerical columns in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing numerical data.
    target : str, optional
        The name of the target column, if one exists.

    Returns:
    --------
    Dict - Dictionary containing two subdictionaries: z_score_outliers and iqr_outliers.
    """
    z_score_outliers = {}
    iqr_outliers = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"{'_'*100}")
            print(f'Attribute: {col}')
            if target and col == target:
                print("*"*27, "\n****** TARGET COLUMN ******\n", "*"*27)
            print(f"{'_'*100}")

            # Get stats
            print(df[col].describe())
            print()

            # If discrete checks counts
            if is_discrete(df, col):
                check_frequency(df, col)
                print()

            # Check for outliers
            z_score_outliers.update(detect_outliers_zscore(df, col))
            print()
            iqr_outliers.update(detect_outliers_iqr(df, col))
            print()

            # Plots
            plot_column(df, col)
            plot_boxplot(df, col)
        else:
            print(f"Skipping non-numerical column: {col}")

    return {"z_score_outliers": z_score_outliers, "iqr_outliers": iqr_outliers}


################################################################################################################################
# END
################################################################################################################################

if __name__ == "__main__":
    pass