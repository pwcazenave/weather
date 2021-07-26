import logging
import multiprocessing
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cmocean import cm
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset, num2date
from yaml import safe_load

# Headless matplotlib
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rcParams
from matplotlib.colors import LogNorm


logger = logging.getLogger(__name__)


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


def make_frame(fname, x, y, pressure, rain, temperature, time, locations, overwrite=False):
    fig = plt.figure(figsize=(12, 10))
    projection = ccrs.Mercator()
    ax = plt.axes(projection=projection)

    # Create an animation
    rcParams['mathtext.default'] = 'regular'

    # Make a segmented colour map for the rain.
    rain_cm = plt.get_cmap('Blues', 3)

    # Plots the requested time from the model output
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
        ax.axes.text(0.016, 0.975, time.strftime('%Y-%m-%d %H:%M'), ha='left', va='center', color='k', bbox=bbox_props, zorder=300, transform=ax.transAxes)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=96, transparent=True)
    else:
        logger.debug(f'{fname} already exists and overwrite is {overwrite}')

    plt.close()


def make_video(meta, overwrite=False):
    ds = meta['ds']

    logger.debug('Fetching time')
    time = num2date(ds.variables['XTIME'], ds.variables['XTIME'].units)[1:]

    skip_offset = 8  # skip the hindcast days

    # We can now check whether files exist and if we're overwriting
    logger.debug('Check for existing frames on disk')
    fnames = []
    missing_frames = []
    for i in range(ds.dimensions['Time'].size - skip_offset - 1):
        fname = Path('static', 'dynamic', 'frames', f'frame_{i + 1:02d}.png')
        fname.parent.mkdir(parents=True, exist_ok=True)
        if not fname.exists() or overwrite:
            missing_frames.append(True)
        else:
            missing_frames.append(False)
        fnames.append(fname)

    # If we don't have any missing frames and we're not overwriting, we can just skip all the expensive network stuff
    # and use what we have on disk.
    if not any(missing_frames) and not overwrite:
        logger.info('No missing frames and we are not overwriting; skip out early')
        return

    logger.debug('Fetching rain')
    rain = np.diff(ds.variables['RAINNC'], axis=0)
    # Remove zero rainfall values
    rain = np.ma.masked_values(rain, 0)

    # Convert pressure to millibars.
    logger.debug('Fetching pressure')
    surface_pressure = ds.variables['PSFC'][1:]
    base_pressure = ds.variables['PB'][1:, 0]
    pressure = (surface_pressure - base_pressure) / 100

    logger.debug('Fetching temperature')
    temperature = ds.variables['T2']

    with Path('cities.yaml').open('r') as f:
        locations = safe_load(f)

    # Save the animation frames to disk.
    logger.debug('Animating frames')
    pool = multiprocessing.Pool()
    args = []
    for i in range(ds.dimensions['Time'].size - skip_offset - 1):
        si = skip_offset + i
        args.append((fnames[i], meta['x'], meta['y'], pressure[si], rain[si], temperature[si], time[si], locations, overwrite))
        # make_frame(*args[-1])
    pool.starmap(make_frame, args)
    pool.close()
