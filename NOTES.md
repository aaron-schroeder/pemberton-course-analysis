# Notes

## Sources for Pemberton files

`Pemberton Trail Loop.fit`, `Pemberton Trail Loop.tcx`, `Pemberton Trail Loop.gpx`
  
  Source: Alltrails
  
  Notes: 
  - The GPX file fundamentally represents different data. Seems like it could
    just be different elevation data, but hard to tell. Definitely has artifacts of
    rounding off to nearest ~2 meters - lots of flat spots interspersed with rapid
    jumps between terraces.
  - The FIT and TCX files seem to have identical data. But it's still not great.
    The histograms drawn from various sampling distances look nothing alike.
    Ended up going with the walmsley data (below) for making inferences about grades.

`pemberton-loop-trail.gpx`

  Source: trailrunproject

  Notes:
  - This just looks fundamentally different than the elev profile on their website,
    and the distance looks off. Could be a result of under-sampling a GPS route making
    for shorter calculated distances, but I just think this is shitty data that I don't
    have to trust.


`Pemberton_walmsley.gpx`

  Source: [Walmsley's Strava activity](https://www.strava.com/activities/3083545712/overview)

  Notes:
  - This data looks a lot more reliable. Each loop's profile looks similar. The histograms
    look the same no matter the sampling distance.

## Next steps

Quantify the distribution using conventional statistics for normal-ish shaped distributions.
Eg 
  - X% of samples are above 10% uphill grade
  - X% of samples are between +/- 10% grade
  - Any other ways you might quantify a distribution like this.
    I look at the spread on Quad Rock and it becomes clear that
    there is real terrain variability. I look at Horsetooth vs
    Pemberton and realize there is more than one way for a rolling
    course's profile to look. (Horsetooth had more gain per mi,
    but that isn't exactly obvious from the distribution).