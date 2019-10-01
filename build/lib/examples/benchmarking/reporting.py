import pandas as pd

__author__ = 'willmcginnis'


if __name__ == '__main__':
    df = pd.read_csv('peak_mem.csv')
    df = df.sort_values(by='peak_mem(MiB)', ascending=True)
    df = df.set_index(keys='encoder', drop=True)

    # normalize memory by control
    control_mib = df.loc['control', 'peak_mem(MiB)']
    df['memory_factor (smaller better)'] = (df['peak_mem(MiB)'] - control_mib) / control_mib + 1

    # normalize compression
    df['compression (smaller better)'] = df['final_df_size(MB)'] / df['initial_df_size(MB)']

    df = df.reindex(columns=['dataset', 'version', 'memory_factor (smaller better)', 'compression (smaller better)'])
    print(df)