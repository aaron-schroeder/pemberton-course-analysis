from genericpath import samestat
import itertools
import math

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import pandas_xyz
# import power

from app import read_csv, sample_dist
# from reader import read_csv  # (or whatever)


df = read_csv('data/horsetooth.csv')

# print(df)

# split df into one-mile sub-dfs resampled every 50 meters

mileage = df['distance'].max() / 1609.34
bounds_lo = range(math.ceil(mileage))
bounds_hi = itertools.chain(range(1, math.ceil(mileage)), [mileage])

interp_fn = interp1d(df['distance'], df['elevation'], kind='linear')

series_list = []

for interval in zip(bounds_lo, bounds_hi):
  lo_m = interval[0] * 1609.34
  hi_m = interval[1] * 1609.34

  n_sample = math.ceil((hi_m - lo_m) / 50) + 1
  
  distance_resample = np.linspace(lo_m, hi_m, n_sample)

  df_sub = pd.DataFrame(dict(
    distance=distance_resample,
    elevation=interp_fn(distance_resample)
  ))

  dx = df_sub['distance'].diff().dropna()
  dy = df_sub['elevation'].diff().dropna()
  grade_dec = (dy / dx)

  # df_interval = pd.concat([dx, dy, grade_dec], axis=1)
  df_interval = pd.DataFrame(dict(dx=dx, dy=dy, grade_dec=grade_dec))
  df_interval.index = pd.IntervalIndex.from_breaks(distance_resample)

  #---------------------------------------------------------------------
  # Minetti adjusted mile pace

  # import power
  from power.algorithms import minetti

  # J/kg/m -> J/kg
  energy_cost = minetti.cost_of_running(df_interval['grade_dec']) * df_interval['dx']
  # print(energy_cost)
  total_energy_cost = energy_cost.sum()

  # How much harder/less hard do I need to run to stay at flat-ground effort?

  # TODO: Fix this. I think it's inverted.
  # factor = minetti.cost_of_running(df_interval['grade_dec']) / minetti.cost_of_running(0.0)
  # print(factor)

  # J/kg/m * m / (J/kg)
  speed_adjustment_factor = (minetti.cost_of_running(0.0) * (hi_m - lo_m)) / total_energy_cost
  # print(speed_adjustment_factor)

  df_interval['minetti_adjustment_factor'] = (minetti.cost_of_running(0.0) * df_interval['dx']) / energy_cost

  #---------------------------------------------------------------------
  # Strava GAP adjusted mile pace

  from power.algorithms import strava

  # TODO: Fix this in power - don't want to specify "values"
  factors = 1 / strava.gap_factor(df_interval['grade_dec'].values)
  # print(factors)
  df_interval['strava_adjustment_factor'] = factors

  #---------------------------------------------------------------------
  # TrainingPeaks NGP adjusted mile pace

  from power.algorithms import trainingpeaks

  # TODO: Fix this in power - don't want to specify "values"
  factors = 1 / trainingpeaks.ngp_factor(df_interval['grade_dec'].values)
  # print(factors)
  df_interval['trainingpeaks_adjustment_factor'] = factors

  #---------------------------------------------------------------------
  # Calculate adjusted speed for each segment based on the speed factors
  
  speed_ms = 1609.34 / 8 / 60  # min/mile -> m/s
  
  times = df_interval[['minetti_adjustment_factor', 'strava_adjustment_factor', 'trainingpeaks_adjustment_factor']].apply(
    lambda series: df_interval['dx'] / (speed_ms * series),
    axis=0
  )

  total_times = times.sum(axis=0)
  # print(pd.to_timedelta(total_times, unit='s'))
  split_speeds_ms = (hi_m - lo_m) / total_times

  # series_list.append(split_speeds_ms)
  # series_list.append(pd.to_timedelta(total_times, unit='s'))
  split_times = pd.to_timedelta(total_times, unit='s')

  #---------------------------------------------------------------------
  # Calculate elevation gain and loss for the segment
  # print(df_sub['elevation'].xyz.z_gain_threshold())  # No Series accessor
  print(df_sub.xyz.z_gain_threshold(threshold=1))
  print(df_sub[::-1].xyz.z_gain_threshold(threshold=1))
  print('')

# df_new = pd.concat(series_list, axis=1)
df_new = pd.DataFrame(series_list)
# print(df_new)

def format(s):
  ts = s.total_seconds()
  hours, remainder = divmod(ts, 3600)
  assert hours == 0
  minutes, seconds = divmod(remainder, 60)
  return f'{int(minutes):02d}:{int(seconds):02d}'

print(df_new.applymap(format))