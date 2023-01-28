import heartandsole
import plotly.graph_objs as go

from common import read_csv, sample_dist


fig = go.Figure()

for method, fname in [
  (heartandsole.Activity.from_tcx, 'data/Pemberton Trail Loop.tcx'), # doesn't work with TCX course file
  (heartandsole.Activity.from_fit, 'data/Pemberton Trail Loop.fit'),  # doesn't work with fit course file
  (heartandsole.Activity.from_gpx, 'data/Pemberton Trail Loop.gpx'),
  (heartandsole.Activity.from_gpx, 'data/pemberton-loop-trail.gpx'),
]:
  act = method(fname)

  # Distances are not included in GPX files; calculate them from lat/lon.
  if not act.has_streams('distance'):
    act.distance.records_from_position(inplace=True)

  df = act.records

  fig.add_trace(go.Scatter(
    x=df['distance'],
    y=df['elevation'],
    mode='markers',
    name=fname.split('.')[-1]
  ))

fig.show()
# hist.write_image('images/hist.jpeg')