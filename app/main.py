#!/usr/bin/env python
"""
Get a weather forecast from PML's WRF outputs.

Pierre Cazenave - pwcazenave@gmail.com

ChangeLog
    06/07/2021 First release.

"""

# TODO:
#   - Pick a location on a map and get the weather there

import os
import logging
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path

import flask
import flask_wtf
from flask_apscheduler import APScheduler

from cmocean import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from dateutil.relativedelta import relativedelta
from yaml import safe_load
from api import api

# Headless matplotlib
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams

host = os.environ.get('HOST', '127.0.0.1')
port = int(os.environ.get('PORT', 8000))
debug = 'DEBUG' in os.environ
use_reloader = os.environ.get('USE_RELOADER', '1') == '1'
api_key = os.environ.get('API_KEY')

root_logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-15s %(levelname)-4s %(message)s', '%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
root_logger.addHandler(flask.logging.default_handler)
if debug:
    root_logger.setLevel(logging.DEBUG)
else:
    root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.info('Starting app')

app = flask.Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = os.urandom(32)
app.register_blueprint(api, url_prefix='/api/v1')

scheduler = APScheduler()
scheduler.init_app(app)

# Configure CSRF protection.
csrf = flask_wtf.csrf.CSRFProtect(app)
app.config['SECRET_KEY'] = os.urandom(32)
csrf.init_app(app)

# Remove unnecessary whitespace in the rendered HTML
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True


def get_box(lon, lat):
    """ Get the extents for the current WRF grid. """

    west = lon[0, :].min()
    east = lon[-1, :].max()
    south = lat[0, :].max()
    north = lat[-1, :].min()

    return west, east, south, north


def closest_point(target_x, target_y, longitude, latitude):
    """
    Find the closest WRF grid point (longitude, latitude) to target_x and target_y.

    """
    pos_idx = np.argmin(np.sqrt((longitude - target_x)**2 + (latitude - target_y)**2))
    x_idx, y_idx = np.unravel_index(pos_idx, longitude.shape)

    return x_idx, y_idx


def fix_range(values, nmin, nmax):
    """
    Fix the range of the given values to the range nmin-nmax.

    """
    a = values.min()
    b = values.max()
    if a == b:
        return a
    else:
        return (((nmax - nmin) * (values - a)) / (b - a)) + nmin


def add_cities(ax, cities, longitude, latitude, temperature):
    """
    Add the given cities to the given axes. Plot as coloured boxes.

    """
    # Set 1 to be equal to 30 Celsius, which will be the top of the colour bar
    norm_temp = temperature / 30
    colours = plt.get_cmap(cm.thermal)
    for city in cities:
        lat, lon = cities[city]
        xidx, yidx = closest_point(lon, lat, longitude, latitude)
        temp = temperature[yidx, xidx]
        # Make the background temperature in the range 0-255
        bbox = {'boxstyle': 'round', 'pad': 0.3, 'facecolor': colours(fix_range(norm_temp[yidx, xidx], 0, 255)), 'lw': 0, 'alpha': 0.75}
        fc = 'k'
        if temp < 16:
            fc = 'w'
        ax.text(lon, lat, f'{temp:.1f}', ha='center', va='center', color=fc, size=10, bbox=bbox, zorder=250, transform=ccrs.PlateCarree())


@app.route('/')
def slash():
    return flask.redirect(flask.url_for('map'))


@app.route('/weather/')
@app.route('/weather/<count>')
def get_weather_frame(count=1):
    # Return the count'th frame png
    frame_name = Path('static', 'dynamic', 'frames', f'frame_{int(count):02d}.png')
    if frame_name.exists():
        return flask.send_file(frame_name, mimetype='image/png')
    else:
        return flask.Response(status=404)


@app.route('/map')
def create_map():
    meta = get_current_forecast_metadata()
    kwargs = {'west': meta['west'],
              'east': meta['east'],
              'south': meta['south'],
              'north': meta['north'],
              'api_key': api_key}
    # Make sure we have the frames for today
    make_video(meta, overwrite=False)

    return flask.render_template('map.html', **kwargs)


def get_current_forecast_metadata():
    # Fetch the latest dataset from PML
    today = datetime.now()
    year = f'{today:%Y}'
    month = f'{today:%m}'
    day = f'{today:%d}'
    run_day = today - relativedelta(days=2)
    meta = {}
    worked = False
    max_tries = 30
    tries = 0
    while not worked and tries < max_tries:
        meta['url'] = f'https://data.ecosystem-modelling.pml.ac.uk/thredds/dodsC/mycoast-all-files/Model/NORTHWESTSHELF_FORECAST_WIND_002/northwestshelf_forecast_wind_002_hourly_all/{year}/{month}/wrfout_d03_{run_day:%Y-%m-%d}_21_00_00.nc'
        try:
            meta['ds'] = Dataset(meta['url'])
            logger.debug(f'Found forecast for {run_day:%Y-%m-%d}')
            worked = True
        except OSError:
            logger.debug(f'Failed to find forecast for {run_day:%Y-%m-%d}')
            # File might not exists on the thredds server, try an older file
            run_day -= relativedelta(days=1)
            tries += 1

    meta['x'] = meta['ds'].variables['XLONG'][0]
    meta['y'] = meta['ds'].variables['XLAT'][0]
    meta['west'], meta['east'], meta['south'], meta['north'] = get_box(meta['x'], meta['y'])

    return meta


def make_frame(i, x, y, pressure, rain, temperature, time, locations, overwrite=False, skip_offset=0):
    fig = plt.figure(figsize=(12, 10))
    projection = ccrs.Mercator()
    ax = plt.axes(projection=projection)

    # Create an animation
    rcParams['mathtext.default'] = 'regular'

    # Make a segmented colour map for the rain.
    rain_cm = plt.get_cmap('Blues', 3)

    # Plots the requested time from the model output
    fname = Path('static', 'dynamic', 'frames', f'frame_{i + 1:02d}.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    if not fname.exists() or overwrite:
        logger.debug(f'Creating {fname}')
        ax.clear()
        ax.axis('off')
        ax.contour(x, y, pressure, levels=np.arange(0, 1000, 5), colors=['white'], nchunk=5, transform=ccrs.PlateCarree(), zorder=50)
        cf = ax.contourf(x, y, rain, transform=ccrs.PlateCarree(), linestyles=None, alpha=0.75, zorder=150, cmap=rain_cm, norm=LogNorm())
        cf.set_clim(vmax=2)
        # Convert Kelvin to Celsius here
        add_cities(ax, locations['cities'], x, y, temperature - 273.15)
        # ax.coastlines(zorder=200, color='w', linewidth=1)
        ax.set_extent((x.min(), x.max(), y.min(), y.max()), crs=ccrs.PlateCarree())
        bbox_props = dict(boxstyle='round, pad=0.3', facecolor='w', edgecolor='w', lw=0, alpha=0.5)
        ax.axes.text(0.016, 0.975, time.strftime('%Y-%m-%d %H:%M'),
            ha='left', va='center', color='k', bbox=bbox_props, zorder=300, transform=ax.transAxes)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=96, transparent=True)
    else:
        logger.debug(f'{fname} already exists and overwrite is {overwrite}')

    plt.close()


def make_video(meta, overwrite=False):
    ds = meta['ds']

    rain = np.diff(ds.variables['RAINNC'], axis=0)
    # Convert pressure to millibars.
    surface_pressure = ds.variables['PSFC'][1:]
    base_pressure = ds.variables['PB'][1:, 0]
    pressure = (surface_pressure - base_pressure) / 100
    # Remove zero rainfall values
    rain = np.ma.masked_values(rain, 0)
    temperature = ds.variables['T2']
    time = num2date(ds.variables['XTIME'], ds.variables['XTIME'].units)[1:]

    with Path('cities.yaml').open('r') as f:
        locations = safe_load(f)

    skip_offset = 8  # skip the hindcast days

    # Save the animation frames to disk.
    pool = multiprocessing.Pool()
    args = []
    for i in range(ds.dimensions['Time'].size - skip_offset - 1):
        si = skip_offset + i
        args.append((i, meta['x'], meta['y'], pressure[si], rain[si], temperature[si], time[si], locations, overwrite))
        # make_frame(*args[-1])
    pool.starmap(make_frame, args)
    pool.close()

    # for i in :
    #     animate(i, overwrite=overwrite)


@scheduler.task('cron', id='make_video', day='*', hour=2, minute=30)
def today_video():
    # Create frames for the most recent model run
    meta = get_current_forecast_metadata()
    make_video(meta, overwrite=True)


@app.context_processor
def inject_today_date():
    """ Makes 'today_date' and 'today_year' be usable in the jinja templates """
    return {'today_date': datetime.now().strftime('%Y-%m-%d'), 'today_year': datetime.now().year}


@app.context_processor
def utility_functions():
    def print_in_console(message):
        print(str(message))

    return dict(mdebug=print_in_console)


@app.errorhandler(flask_wtf.csrf.CSRFError)
def handle_csrf_error(e):
    return flask.jsonify({'success': False, 'error': e.description}), 400


def main():
    app.run(host=host,
            port=port,
            debug=debug,
            use_reloader=use_reloader,
            #ssl_context='adhoc',
            extra_files=['./app/templates/index.html',
                         './app/templates/map.html',
                         './app/templates/favicon.ico',
                         './app/static/css/style.css'])


if __name__ == '__main__':

    logger.info('Starting scheduler')
    scheduler.start()
    logger.info('Starting main')
    main()
