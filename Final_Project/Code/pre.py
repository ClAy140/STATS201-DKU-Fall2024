import os
import ipdb
# ipdb.set_trace()
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

dataU = pd.read_csv('wind_gen_cf_2020.csv', parse_dates=['datetime'], index_col='datetime')
dataC = pd.read_csv('Wind_capacity_99MW.csv', parse_dates=['Time(year-month-day h:m:s)'], index_col='Time(year-month-day h:m:s)')

config = pd.read_csv('eia_wind_configs.csv')
print(dataU.head())