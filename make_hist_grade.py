import plotly.express as px
import plotly.graph_objs as go

from common import read_csv, sample_dist


# fname = 'data/horsetooth.csv'
# fname = 'data/quadrock.csv'
# fname = 'data/pemberton.csv'
fname = 'data/pemberton_walmsley.csv'
# fname = 'data/redhot.csv'

# act = heartandsole.Activity.from_csv(fname)
df = read_csv(fname)

# print(df)

# Make a histogram of the point-to-point spacing
# hist = px.histogram(df.diff(), x='elevation', nbins=20)
hist = go.Figure(layout=dict(title=fname))

for dx in [5.0, 10.0, 25.0, 50.0]:
  df_sub = sample_dist(df, len_sample=dx)
  # df_sub = act.resample_records(len_sample=dx)
  # print(df_sub)

  # Reverse direction for grades (loop runs CW, this file runs CCW)
  grades = -df_sub['elevation'].diff() / df_sub['distance'].diff()
  # grades = act_sub.grade.dy_dx()
  # grades = df_sub.xyz.dy_dx()
  # print(grades)

  hist.add_trace(go.Histogram(
    x=grades,
    # nbinsx=20,
    xbins=dict(
      start=-0.25 - 0.025/2,
      end=0.25 + 0.25/2,
      size=0.025
    ),
    autobinx=False,
    histnorm='percent',
    name=dx,
    # marginal='rug',
  ))

  # Add a vertical line representing the median value
  # med = grades.median()
  # print(f'median = {med:.1f}')
  # hist.add_vline(x=med, line_width=2, line_dash='dash', line_color='green')

# Overlay both histograms
# hist.update_layout(barmode='overlay')
# Reduce opacity to see all histograms
hist.update_traces(opacity=0.5)

hist.update_layout(hovermode='x')

hist.update_xaxes(range=[-0.5, 0.5])

hist.show()
# hist.write_image('images/hist.jpeg')