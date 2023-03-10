# Pemberton 50k course analysis dashboard

## Viewing the dashboard

To view the dashboard itself, first install the dependencies:
```
>>> pip install -r requirements.txt
```
Then run the dashboard and navigate to the link where it is hosted locally
on your machine:
```
>>> python app.py
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
```

## Extra: Histogram generation

To see how I generated the histogram of the file's point-to-point distances,
check out `make_hist.py`.

## Extra: Data wrangling

If you are interested in how the data wrangling happens, you'll need
to install the extra requirements:
```
>>> pip install -r requirements_wrangling.txt
```
Then, you can run the python script which converts a GPX file to a
pandas-ready CSV file, which will appear in `data/`.
```
python wrangling.py
```