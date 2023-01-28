"""wrangling.py"""
import heartandsole


# fname_in = 'data/Horsetooth Half Marathon.gpx'
# fname_out = 'data/horsetooth.csv'

# # fname_in = 'data/Pemberton Trail Loop.tcx' 
# fname_in = 'data/Pemberton Trail Loop.fit'
# # fname_in = 'data/Pemberton Trail Loop.gpx'
# # fname_in = 'data/pemberton-loop-trail.gpx'  # TrailRunProject - short!
# fname_out = 'data/pemberton.csv'

fname_in = 'data/Pemberton_walmsley.gpx'
fname_out = 'data/pemberton_walmsley.csv'

# fname_in = 'data/Quad Rock 25 Mile Original Route.tcx'
# fname_out = 'data/quadrock.csv'

# fname_in = 'data/red-hot-55k.gpx'
# fname_out = 'data/redhot.csv'

if fname_in.lower().endswith('tcx'):
  act = heartandsole.Activity.from_tcx(fname_in)
elif fname_in.lower().endswith('fit'):
  act = heartandsole.Activity.from_fit(fname_in)
elif fname_in.lower().endswith('gpx'):
  act = heartandsole.Activity.from_gpx(fname_in)

# If distances are not included in the file, calculate them from lat/lon.
if not act.has_streams('distance'):
  print('adding distance coordinates')
  act.distance.records_from_position(inplace=True)

# TODO: Determine if I want to save any grade values in-file.
# if act.has_streams('elevation'):
#   import numpy as np  # to perform a central diff method, not simple differences between pts.
  
#   # act.records['grade'] = 100.0 * act.records['elevation'].diff() / act.records['distance'].diff()
#   act.records['grade'] = 100.0 * act.records.xyz.z_smooth_distance().diff() / act.records['distance'].diff()
#   # act.records['grade'] = 100.0 * np.gradient(act.records.xyz.z_smooth_distance(), act.records['distance'])
#   # act.records['grade'] = 100.0 * np.gradient(act.records['elevation'], act.records['distance'])

# Save DF as a CSV file
act.records.to_csv(fname_out)