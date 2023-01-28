"""Display data from a csv Activity file in an interactive dashboard."""
import math
import datetime

from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import numpy as np
import pandas as pd
import pandas_xyz
from scipy.interpolate import interp1d

import common


def create_dash_app():

  app = common.create_dash_app()

  app.layout.children[3].children.append(
    dbc.Col(
      [
        dbc.InputGroup([
          dbc.InputGroupText('Flat HM Time:'),
          dbc.Input(
            id='flat-hours',
            type='number', min=0, max=3,
            step=1, 
            value=1,
          ),
          dbc.InputGroupText(':'),
          dbc.Input(
            id='flat-minutes',
            type='number', min=0, max=59,
            step=1, 
            value=30,
          ),
          dbc.InputGroupText(':'),
          dbc.Input(
            id='flat-seconds',
            type='number', min=0, max=59,
            step=1, 
            value='00',
          ),
        ]),
      ],
      # width=12,
      md=6,
    ),
  )

  # Initialize callbacks
  @app.callback(
    Output('figure', 'figure'),
    Output('stats', 'children'),
    Input('slider', 'value'),
    Input('len-sample', 'value'),
    Input('flat-hours', 'value'),
    Input('flat-minutes', 'value'),
    Input('flat-seconds', 'value'),
    State('activity-data', 'data'),
  )
  def update_fig_and_stats(slider_values, len_sample, hours, minutes, seconds, record_data):
    """Update plot and statistics to desired range and sample interval."""

    if record_data is None or hours is None or minutes is None or seconds is None:
      raise PreventUpdate

    len_sample = len_sample or common.SAMPLE_LEN

    df = pd.DataFrame.from_records(record_data)

    df_sub = common.sample_dist(df, slider_values[0], slider_values[1], len_sample)

    speed_target = 21097.5 / (float(hours) * 3600 + float(minutes) * 60 + float(seconds))
    # speed_target = process_pace_input(hours, minutes, seconds)
    # speed_target = 3.4
    print(speed_target)

    fig = create_fig(df, df_sub, speed_target)

    stats_children = create_stats_table(df_sub, speed_target)

    # FIX this. The highlighted points no longer make sense.
    # fig.update_traces(selectedpoints=df_sub.index)

    return fig, stats_children

  return app


def create_fig(df, df_sub=None, speed_target=None):
  """Generate a plotly figure of an elevation profile.

  Creates a plotly figure for use as `dcc.Graph.figure` representing
  the elevation profile contained in the input DataFrame. The first
  trace represents the raw data. If ``df_sub`` is passed as an input,
  a second trace is added with the sub-sampled elevation profile data,
  including a filled area between each pair of points representing the
  grade between them.

  Args:
    df (pd.DataFrame): A DataFrame representing a recorded activity.
      Each row represents a record, and each column represents a stream
      of data. Assumed to have ``elevation`` and ``distance`` columns.
    df_sub (pd.DataFrame): Optional. A DataFrame representing a subset
      of the data in ``df``, re-sampled at even distance intervals.

  Returns:
    plotly.graph_objects.Figure: figure to be used with `dcc.Graph`
  """
  fig = common.create_fig(df)

  # ----------------------------------------------------------------------
  # Identify adjusted speed with filled regions

  if df_sub is not None:

    x = df_sub['distance']
    y = df_sub['elevation']

    # dydx = 100 * np.diff(y) / np.diff(x)
    df_times = calc_adjusted_speeds(df_sub, speed_target)
    df_speeds = 1 / df_times.div(df_times.index.length, axis=0)
    series_speeds = df_speeds['strava_adjustment_factor']

    # colorscale = 'Picnic'
    # colorscale = 'Tropic'
    # colorscale = 'Jet'
    colorscale = 'Rainbow'

    # speed_min = 1.5
    # speed_max = 4.0
    speed_min = series_speeds.min()
    speed_max = series_speeds.max()

    # Add a dummy trace so the colorbar will appear on the plot.
    fig.add_trace(go.Scatter(
      x=0.5 * (x[:-1] + x[1:]),
      y=[0.5 * fig.layout.yaxis.range[0] for i in range(len(x))],
      mode='markers',
      text=series_speeds,
      hoverinfo='skip',
      marker=dict(
        size=None,
        color=series_speeds,
        # cmin=-25.0,
        # cmax=25.0,
        cmin=speed_min,
        cmax=speed_max,
        colorbar=dict(
          title='Speed',
          orientation='h',
        ),
        colorscale=colorscale,
      ),
      showlegend=False,
      # visible='legendonly',
    ))
    
    from plotly.colors import sample_colorscale

    fill_colors = sample_colorscale(colorscale, (series_speeds-speed_min)/(speed_max-speed_min), colortype='rgb')

    for x_pair, y_pair, fill_color, speed_i in zip(zip(x, x[1:]), zip(y, y[1:]), fill_colors, series_speeds):

      fig.add_trace(go.Scatter(
        x=[x_pair[0], x_pair[0], x_pair[1], x_pair[1]],
        y=[0, y_pair[0], y_pair[1], 0],
        fill='toself',
        # x=x_pair,
        # y=y_pair,
        # fill='tozeroy',
        name=f'Strava: {speed_i:.1f} m/s',
        # x=x_pair,
        # y=y_pair,
        mode='lines',
        # mode='lines+markers',
        # mode='markers',
        # mode=None,
        marker=dict(
          size=2,
        ),
        line=dict(
          # color='gray',
          width=0,
        ),
        fillcolor=fill_color,
        hoveron='fills', # select where hover is active
        showlegend=False,
      ))

    fig.add_trace(go.Scatter(
      x=x,
      y=y,
      name='Re-sampled data',
      # mode='lines',
      # mode='lines+markers',
      mode='markers',
      # mode=None,
      marker=dict(
        size=2,
        color='gray',
      ),
      hovertemplate='%{y:.1f} m at %{x:.1f} m',
    ))

  fig.update_xaxes(dict(tickvals=[1609.34 * (i + 1) for i in range(13)]))

  return fig


def create_stats_table(df_sub, speed_target):
  """Generate a table of statistics about an elevation profile.

  Creates a list for use as `dcc.Div.children` containing a single
  `dbc.Table` with statistics about the elevation profile data in
  ``df_sub``. Start and end locations, total length and elevation gain,
  and average/min/max target pace.

  Args:
    df (pd.DataFrame): A DataFrame representing a recorded activity.
      Each row represents a record, and each column represents a stream
      of data. Assumed to have ``elevation`` and ``distance`` columns.
    df_sub (pd.DataFrame): A DataFrame representing an elevation profile.
      Assumed to have ``elevation`` and ``distance`` columns, with the
      ``distance`` data points evenly spaced.

  Returns:
    list(dbc.Table): children to be used with `dcc.Div`. Just contains
    one table.
  """
  x_1 = df_sub['distance'].iloc[-1]
  x_0 = df_sub['distance'].iloc[0]
  dx = x_1 - x_0

  df_times = calc_adjusted_speeds(df_sub, speed_target)
  df_speeds = 1 / df_times.div(df_times.index.length, axis=0)
  # print(times_df)

  stats_table_header = [
    html.Thead([
      html.Tr([
        html.Th('Distance', colSpan=2),
        html.Th('Elevation', colSpan=2),
        # html.Th('Net elevation difference', rowSpan=2),
      ]),
      html.Tr([
        html.Th('Start'),
        html.Th('End'),
        # html.Th('Length'), 
        html.Th('Gain'),
        html.Th('Loss'),
        # html.Th('Net elevation difference'),
      ])
    ])
  ]

  stats_row = html.Tr([
    html.Td(f"{x_0:.0f} m"),
    html.Td(f"{x_1:.0f} m"),
    # html.Td(f"{dx:.0f} m / {dx/1609.34:.1f} mi"),
    # html.Td(f"{dy:.0f} m / {dy*5280/1609.34:.0f} ft"),
    # html.Td(f"{dx:.0f} m"),
    # html.Td(f"{dy:.0f} m"),
    html.Td(f"{df_sub.xyz.z_gain_threshold(threshold=1):.1f} m"),
    html.Td(f"{df_sub[::-1].xyz.z_gain_threshold(threshold=1):.1f} m"),
    # html.Td(f"{100*dy/dx:.1f}%"),
    # html.Td(f"{100*dydx.max():.1f}%"),
    # html.Td(f"{100*dydx.min():.1f}%"), 
  ])

  stats_table_body = [html.Tbody([stats_row])]

  # pace_table_header = [
  #   html.Thead([
  #     html.Tr([
  #       html.Th('Adjustment algorithm', rowSpan=2),
  #       html.Th(f'Pace (sampled every {dx/len(df_sub):.1f} m)', colSpan=3),
  #       html.Th('Time', rowSpan=2)
  #     ]),
  #     html.Tr([
  #       html.Th('Avg.'),
  #       html.Th('Max.'),
  #       html.Th('Min.'),
  #     ])
  #   ])
  # ]

  pace_table_header = [
    html.Thead([
      html.Tr([
        html.Th(colSpan=2),
        html.Th('Minetti'),
        html.Th('Strava'),
        html.Th('TrainingPeaks'),
        # html.Th('Flat'),
      ]),
    ])
  ]

  # pace_rows = [
  #   html.Tr([
  #     html.Td(col.split('_')[0]),
  #     # html.Td(f"{(df_times.index.right - df_times.index.left).to_series().sum() / df_times[col].sum():.2f}"),
  #     # html.Td(f"{df_speeds[col].max():.2f}"),
  #     # html.Td(f"{df_speeds[col].min():.2f}"),
  #     html.Td(f"{speed_to_pace((df_times.index.right - df_times.index.left).to_series().sum() / df_times[col].sum())}"),
  #     html.Td(f"{speed_to_pace(df_speeds[col].max())}"),
  #     html.Td(f"{speed_to_pace(df_speeds[col].min())}"),
  #     html.Td(f"{format_time(pd.to_timedelta(df_times[col].sum(), unit='s'))}"),
  #   ])
  #   for col in df_speeds.columns
  # ]

  pace_rows = [
    html.Tr(
      [html.Td('Pace', rowSpan=3), html.Td('Average')] +
      [
        html.Td(f"{speed_to_pace((df_times.index.length).to_series().sum() / df_times[col].sum())}")
        for col in df_times.columns
      ]
    ),
    html.Tr(
      [html.Td('Max')] +
      [
        html.Td(f"{speed_to_pace(df_speeds[col].max())}")
        for col in df_times.columns
      ]
    ),
    html.Tr(
      [html.Td('Min')] +
      [
        html.Td(f"{speed_to_pace(df_speeds[col].min())}")
        for col in df_times.columns
      ]
    ),
    html.Tr(
      [html.Td('Time', rowSpan=15), html.Td('Total')] +
      [
        html.Td(f"{format_time(pd.to_timedelta(df_times[col].sum(), unit='s'))}")
        for col in df_times.columns
      ]
    ),
  ]


  # janky, hacky...
  t_pre_prev = 0.0
  split_times = []
  for i in range(13):
    m_split_pre = 1609.34 * i
    m_split = 1609.34 * (i + 1)
    t = df_times.loc[m_split]
    t_pre = t * (m_split - t.name.left) / (t.name.length)
    
    split_time = pd.to_timedelta(df_times.loc[m_split_pre:m_split].iloc[:-1].sum() + t_pre - t_pre_prev, unit='s')
    split_times.append(split_time)

    t_pre_prev = t_pre

  split_times.append(
    pd.to_timedelta(df_times.loc[m_split:].sum() - t_pre_prev, unit='s')
  )  
  
  splits_df = pd.DataFrame(split_times)
  
  # print(splits_df)
  # print(splits_df.sum())  # should == the race times calc'd elsewhere

  # TODO: Find a way to include cumulative time, if I don't handle it elsewhere.
  for ix, row in splits_df.iterrows():
    pace_rows.append(html.Tr(
      [html.Td(f"Mile {ix + 1}")] +
        [
          html.Td(f"{format_pace(pd.to_timedelta(val, unit='s'))}")
          for val in row.values
        ]
      ),
    )

  # pace_rows.append(html.Tr([
  #   html.Td('flat'),
  #   html.Td(f"{speed_to_pace(speed_target)}"),
  #   html.Td(f"{speed_to_pace(speed_target)}"),
  #   html.Td(f"{speed_to_pace(speed_target)}"),
  #   html.Td(f"{format_time(pd.to_timedelta((df_times.index.right - df_times.index.left).to_series().sum() / speed_target, unit='s'))}"),
  # ]))

  pace_table_body = [html.Tbody(pace_rows)]

  return [
    dbc.Table(
      stats_table_header + stats_table_body,
      bordered=True,
      style={'text-align': 'center'},
    ),
    dbc.Table(
      pace_table_header + pace_table_body,
      bordered=True,
      style={'text-align': 'center'},
    ),
  ]


def calc_adjusted_speeds(df, speed_target):
  # interp_fn = interp1d(df['distance'], df['elevation'], kind='linear')

  # lo_m = interval[0] * 1609.34
  # hi_m = interval[1] * 1609.34

  # n_sample = math.ceil((hi_m - lo_m) / 50) + 1
  
  # distance_resample = np.linspace(lo_m, hi_m, n_sample)

  # df_sub = pd.DataFrame(dict(
  #   distance=distance_resample,
  #   elevation=interp_fn(distance_resample)
  # ))

  dx = df['distance'].diff().dropna()
  dy = df['elevation'].diff().dropna()
  grade_dec = (dy / dx)

  # df_interval = pd.concat([dx, dy, grade_dec], axis=1)
  df_interval = pd.DataFrame(dict(dx=dx, dy=dy, grade_dec=grade_dec))
  df_interval.index = pd.IntervalIndex.from_breaks(df['distance'])

  #---------------------------------------------------------------------
  # Minetti adjusted mile pace

  # import power
  from power.algorithms import minetti

  # J/kg/m -> J/kg
  energy_cost = minetti.cost_of_running(df_interval['grade_dec']) * df_interval['dx']
  # print(energy_cost)
  # total_energy_cost = energy_cost.sum()

  # How much harder/less hard do I need to run to stay at flat-ground effort?

  # TODO: Fix this. I think it's inverted.
  # factor = minetti.cost_of_running(df_interval['grade_dec']) / minetti.cost_of_running(0.0)
  # print(factor)

  # J/kg/m * m / (J/kg)
  # speed_adjustment_factor = (minetti.cost_of_running(0.0) * (hi_m - lo_m)) / total_energy_cost
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
  
  # speed_target = 1609.34 / 8 / 60  # min/mile -> m/s
  
  times = df_interval[['minetti_adjustment_factor', 'strava_adjustment_factor', 'trainingpeaks_adjustment_factor']].apply(
    lambda series: df_interval['dx'] / (speed_target * series),
    axis=0
  )

  total_times = times.sum(axis=0)
  # print(pd.to_timedelta(total_times, unit='s'))
  # split_speeds_ms = (hi_m - lo_m) / total_times

  # series_list.append(split_speeds_ms)
  # series_list.append(pd.to_timedelta(total_times, unit='s'))
  split_times = pd.to_timedelta(total_times, unit='s')

  #---------------------------------------------------------------------
  # Calculate elevation gain and loss for the segment
  # print(df_sub['elevation'].xyz.z_gain_threshold())  # No Series accessor
  # print(df.xyz.z_gain_threshold(threshold=1))
  # print(df[::-1].xyz.z_gain_threshold(threshold=1))
  # print('')

  return times


def speed_to_pace(v):
  seconds_per_mile = 1609.34 / v

  return format_pace(pd.to_timedelta(seconds_per_mile, unit='s'))


def format_time(td):
  ts = td.total_seconds()
  hours, remainder = divmod(ts, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'


def format_pace(td):
  ts = td.total_seconds()
  hours, remainder = divmod(ts, 3600)
  assert hours == 0
  minutes, seconds = divmod(remainder, 60)
  return f'{int(minutes):02d}:{int(seconds):02d}'


if __name__ == '__main__':
  app = create_dash_app()
  app.run_server(
    # debug=True
  )