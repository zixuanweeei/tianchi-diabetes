# coding: utf-8

import glob
from datetime import datetime as dt
import pandas as pd

filenames = glob.glob('../submission/result*.csv')
print(filenames)
result = []
for filename in filenames:
    result.append(pd.read_csv(filename, header=None))
result = pd.concat(result, axis=1)
result = result.mean(axis=1)
result.to_csv('../submission/re_melt_{0}.csv'.format(dt.now().strftime('%Y%m%d_%H%M%S')),
                        index=False, header=False)
