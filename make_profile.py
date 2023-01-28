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

fig = go.Figure(
    go.Scatter(
    x=df['distance'],
    y=df['elevation'],
    mode='lines+markers',
  ),
  layout=dict(title=fname),
)

for dx in [5.0, 10.0, 25.0, 50.0]:
  df_sub = sample_dist(df, len_sample=dx)
  # print(df_sub)

  fig.add_trace(go.Scatter(
    x=df_sub['distance'],
    y=df_sub['elevation'],
    mode='markers',
    name=dx,
  ))

  # Add a vertical line representing the median value
  # med = grades.median()
  # print(f'median = {med:.1f}')
  # hist.add_vline(x=med, line_width=2, line_dash='dash', line_color='green')

# Reduce opacity to see all histograms
# hist.update_traces(opacity=0.5)

# hist.update_layout(hovermode='x')

fig.show()
# hist.write_image('images/hist.jpeg')