import math

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

SAMPLE_LEN = 100.0


def create_dash_app():
  app = Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
      # 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.8.3/plotly-mapbox.js',
      'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.8.3/plotly-mapbox.min.js',
    ]
  )

  df = read_csv('data/horsetooth.csv')

  d_min = df['distance'].min()
  d_max = df['distance'].max()

  app.layout = dbc.Container(
    [
      dcc.Loading(
        id='loading-1',
        type='default',
        children=html.Div(
          id='loading-output-1',
          children=[
            dcc.Graph(
              id='figure', 
              figure=create_fig(df),
            ),
          ]
        )
      ),
      dcc.Store( # data streams
        id='activity-data',
        data=df.to_dict('records'),
      ),
      dbc.Row(
        [
          dbc.Col(
            dcc.RangeSlider(
              id='slider',
              # min=0,
              # max=len(df) - 1,
              min=d_min,
              max=d_max,
              # step=1,
              # step=None,
              marks={
                1000 * i: f'{i}k' for i in range(math.ceil(d_max / 1000))
              },
              # value=[0, len(df) - 1],
              value=[d_min, d_max],
              # value=[df['distance'][0], df['distance'].iloc[-1]],
              allowCross=False,
              tooltip=dict(
                placement='bottom',
                always_visible=False
              ),
              # style={'padding': '0px 0px 25px'},
              className='px-1',
            ),
            width=12,
            className='mb-3',
          ),
        ],
        justify='center'
      ),
      dbc.Row([
        dbc.Col(
          [
            dbc.InputGroup([
              dbc.InputGroupText('Sample elevation every'),
              dbc.Input(
                id='len-sample',
                type='number', min=5, max=100,
                step=5,
                value=100,
              ),
              dbc.InputGroupText('meters'),
            ]),
            dbc.FormText('Choose a value between 5 and 100.'),
          ],
          # width=12,
          md=6,
        ),
      ]),
      dcc.Loading(
        id='loading-2',
        type='default',
        children=html.Div(
          id='loading-output-2',
          children=[
            html.Hr(),
            # TODO: Change to just 'Statistics'?
            html.H2('Hill Statistics'),
            html.Div(id='stats')
          ],
        ),
      ),
    ],
    id='dash-container',
    # fluid=True,
  )

  return app


def read_csv(fname):
  return pd.read_csv(fname,
    index_col=[0],
    # header=[0, 1], 
    # parse_dates=[2]
  )


def sample_dist(df, bound_lo=None, bound_hi=None, len_sample=5.0):
  """Subsample elevation data in evenly-spaced distance intervals.
  
  First, an evenly-spaced array of distance values is generated,
  spanning from ``bound_lo`` (or the lowest distance value in ``df``)
  to ``bound_hi`` (or the highest distance value in ``df``). The spacing
  is as close to ``len_sample`` as possible without exceeding it.

  Args:
    df (pd.DataFrame): A DataFrame representing a recorded activity.
      Each row represents a record, and each column represents a stream
      of data. Assumed to have ``elevation`` and ``distance`` columns.
    bound_lo (float): Optional. The desired lower bound of the returned
      sub-sampled DataFrame's ``distance`` column.
    bound_hi (float): Optional. The desired upper bound of the returned
      sub-sampled DataFrame's ``distance`` column.
    len_sample (float): The maximum desired point-to-point spacing, in
      meters. Default 5.0.
      
  Returns:
    pd.DataFrame: a subset of the input DataFrame, resampled at evenly-
    spaced distance coordinates. Contains only ``distance`` and 
    ``elevation`` columns.
  """
  # TODO: Verify bound_lo and bound_hi make sense

  bound_lo = bound_lo or df['distance'].iloc[0]  
  bound_hi = bound_hi or df['distance'].iloc[-1]

  n_sample = math.ceil(
    (bound_hi - bound_lo) / len_sample
  ) + 1

  distance_ds = np.linspace(
    bound_lo,
    bound_hi, 
    n_sample
  )

  interp_fn = interp1d(df['distance'], df['elevation'], 
    kind='linear'
    # kind='quadratic'
    # kind='cubic'
  )

  return pd.DataFrame(dict(
    distance=distance_ds,
    elevation=interp_fn(distance_ds)
  ))


def create_fig(df):
  fig = go.Figure(layout=dict(
    xaxis=dict(
      range=[df['distance'].min(), df['distance'].max()],
      showticklabels=False,
    ),
    yaxis=dict(
      range=[0.9 * df['elevation'].min(), 1.1 * df['elevation'].max()],
      showticklabels=False
    ),
    margin=dict(
      b=0,
      # t=0,
      r=0,
      l=0,
    ),
    showlegend=False,
  ))

  return fig


def make_raw_data_trace(df):
  # Show the raw data so we can tell what we're sampling from
  # TODO: Verify that the figure still looks ok when this is added before other data.
  return go.Scatter(x=df['distance'], y=df['elevation'], name='Raw data', 
    # mode='lines+markers',
    # mode='lines',
    # line=dict(
      # color='black',
      # width=1,
    # ),
    mode='markers',
    marker=dict(
      color='black',
      size=4,
    ),
    legendrank=1,  # show at top of list in legend
    hovertemplate='%{y:.1f} m at %{x:.1f} m',
  )