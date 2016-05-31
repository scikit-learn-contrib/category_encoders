"""

.. module:: utils
  :synopsis: generic helper functions shared by many encoders
  :platform:

"""

__author__ = 'willmcginnis'


def get_obj_cols(df):
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object':
            obj_cols.append(df.columns.values[idx])

    return obj_cols
