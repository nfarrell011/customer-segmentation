import pandas as pd
import numpy as np


def update_coef_log(coef_log_name, coef, attr_list, name):

    df = pd.read_csv(coef_log_name)

    coef_dict = dict(zip(attr_list, *coef.astype(float)))

    coef_list = []
    for index, row in df.iterrows():
        try:
            coef_list.append(coef_dict[row['attr']])
        except KeyError:
                coef_list.append(np.nan)

    df[name] = coef_list

    df.to_csv(coef_log_name, index=False)


if __name__ == '__main__':
    pass
