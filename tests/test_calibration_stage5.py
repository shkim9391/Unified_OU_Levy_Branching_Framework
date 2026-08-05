import pandas as pd
from oulb.calibration import calibrate_from_interval_table

def test_calibration():
 x=calibrate_from_interval_table(pd.DataFrame({'displacement_hd':[1,2,3]})); assert x['n_intervals']==3 and x['displacement_median']==2
