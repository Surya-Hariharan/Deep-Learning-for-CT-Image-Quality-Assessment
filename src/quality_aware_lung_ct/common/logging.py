"""CSV run logging.

Migrated unmodified from the author's `utils/common.py:LogWriter`.
"""

import os
import time

import pandas as pd


class LogWriter:
    ''' Log Writer Based on Pandas '''

    def __init__(self, save_dir: str, prefix: str = None):
        ''' Args:
        * `save_dir`: save place of log files.
        * `prefix`: prefix-name of the log file.
        '''
        self.data = pd.DataFrame()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        now = time.strftime("%y%m%d%H%M", time.localtime())
        fname = f"{prefix}-{now}.csv" if prefix else f"{now}.csv"
        self.path = os.path.join(save_dir, fname)

    def add_row(self, data: dict):
        temp = pd.DataFrame(data, index=[0])
        self.data = pd.concat([self.data, temp], ignore_index=True)

    def save(self):
        self.data.to_csv(self.path, index=False)
        print("SAVE runtime logs to `%s`..." % self.path)
