import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
def prof(dataset,path,cat):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cols=dataset.select_dtypes(include=numerics).columns.values
    profile=ProfileReport(dataset)
    profile.to_file(output_file=path)
    if cat!=1:
        return cols
    else:
        return dataset.columns.values
