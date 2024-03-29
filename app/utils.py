import logging
import multiprocessing
from datetime import datetime
from pathlib import Path

# Headless matplotlib
import matplotlib as mpl
mpl.use('Agg')
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cmocean import cm
from dateutil.relativedelta import relativedelta
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from netCDF4 import Dataset, num2date
from yaml import safe_load

# pylint: disable=invalid-name, wrong-import-position, too-many-arguments


logger = logging.getLogger(__name__)


def get_box(lon, lat):
    """ Get the extents for the current WRF grid. """

    west = lon[0, :].min()
    east = lon[-1, :].max()
    south = lat[0, :].max()
    north = lat[-1, :].min()

    return west, east, south, north


def get_ncvars(source, map_type):
    """
    Return a dictionary of the variables for the given model source and type, along with the key dimension names.

    """
    if source == 'pml':
        if map_type == 'atmosphere':
            ncvars = {'time': 'XTIME',
                      'rain': 'RAINNC',
                      'temperature': 'T2',
                      'surface_pressure': 'PSFC',
                      'base_pressure': 'PB',
                      'u': 'U10',
                      'v': 'V10'}
            dims = {'time': 'Time'}
        elif map_type == 'ocean':
            ncvars = {'time': 'time',
                      'u': 'u',
                      'v': 'v',
                      'temperature': 'temp',
                      'salinity': 'salinity'}
            dims = {'time': 'time'}
    elif source == 'gfs':
        if map_type == 'atmosphere':
            ncvars = {'time': 'time',
                      'rain': 'crainsfc',
                      'temperature': 'tmpsfc',
                      'surface_pressure': 'pressfc'}
            dims = {'time': 'time'}

    return ncvars, dims


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
        bbox = {'boxstyle': 'round',
                'pad': 0.3,
                'facecolor': colours(fix_range(norm_temp[yidx, xidx], 0, 255)),
                'lw': 0,
                'alpha': 0.75}
        fc = 'k'
        if temp < 16:
            fc = 'w'
        ax.text(lon, lat, f'{temp:.1f}', ha='center', va='center', color=fc, size=10, bbox=bbox, zorder=250,
                transform=ccrs.PlateCarree())


def get_current_forecast_metadata(source='pml', map_type='atmosphere'):
    """
    Fetch the current forecast metadata for the given model source and type.

    """
    logger.info(f'Fetching {source} {map_type} forecast metadata')

    if source == 'gfs' and map_type == 'ocean':
        raise ValueError('Ocean maps not yet supported for GFS source data')

    if source == 'pml':
        # Fetch the latest dataset from PML
        today = datetime.utcnow()
        year = f'{today:%Y}'
        month = f'{today:%m}'
        run_day = today - relativedelta(days=2)
        meta = {}
        worked = False
        max_tries = 30
        tries = 0
        while not worked and tries < max_tries:
            if map_type == 'atmosphere':
                meta['url'] = f'https://data.ecosystem-modelling.pml.ac.uk/thredds/dodsC/mycoast-all-files/Model/NORTHWESTSHELF_FORECAST_WIND_002/northwestshelf_forecast_wind_002_hourly_all/{year}/{month}/wrfout_d03_{run_day:%Y-%m-%d}_21_00_00.nc'
            elif map_type == 'ocean':
                meta['url'] = 'https://data.ecosystem-modelling.pml.ac.uk/thredds/dodsC/mycoast-all-files/Model/CHANNEL_FORECAST_PHY_001/channel_forecast_phy_001_hourly_t_s_u_v_ssh/today_forecast/today_forecast.nc'

            try:
                meta['ds'] = Dataset(meta['url'])
                logger.debug(f'Found forecast for {run_day:%Y-%m-%d}')
                worked = True
            except OSError:
                logger.debug(f'Failed to find forecast for {run_day:%Y-%m-%d}')
                # File might not exists on the thredds server, try an older file
                run_day -= relativedelta(days=1)
                tries += 1

        if map_type == 'atmosphere':
            meta['x'] = meta['ds'].variables['XLONG'][0]
            meta['y'] = meta['ds'].variables['XLAT'][0]
        elif map_type == 'ocean':
            meta['x'] = meta['ds'].variables['lon']
            meta['y'] = meta['ds'].variables['lat']
            meta['x'], meta['y'] = np.meshgrid(meta['x'], meta['y'])
    elif source == 'gfs':
        # Fetch the latest forecast from GFS
        today = datetime.utcnow()
        ymd = f'{today:%Y%m%d}'
        forecast_delta = []
        forecast_runs = list(range(0, today.hour, 6))
        for i in forecast_runs:
            forecast_delta.append(today - today.replace(hour=i, minute=0, second=0))
        recent_hour = forecast_runs[np.argmin(forecast_delta)]
        hour = f'{recent_hour:02d}'

        meta = {'url': f'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25/gfs{ymd}/gfs_0p25_{hour}z'}
        meta['ds'] = Dataset(meta['url'])
        x = meta['ds'].variables['lon'][:]
        y = meta['ds'].variables['lat'][:]
        meta['x'], meta['y'] = np.meshgrid(x, y)

        # u = d.variables['ugrd10m'][0]
        # v = d.variables['vgrd10m'][0]

    meta['west'], meta['east'], meta['south'], meta['north'] = get_box(meta['x'], meta['y'])

    return meta


def make_atmosphere_frame(fname, x, y, pressure, rain, temperature, locations, overwrite=False):
    """
    Plot an atmospheric model output frame to file `fname' with the given:

        - x, y: coordinate arrays (spherical)
        - pressure, rain, temperature: data arrays (same size as x, y)
        - locations: dictionary with a key 'cities' inside of which is another dictionary with [x, y] positions.

    Optionally pass `overwrite' as True to overwrite existing files, otherwise files are only overwritten if they're
    over 24 hours old. Defaults to False.

    """

    fig = plt.figure(figsize=(12, 10))
    projection = ccrs.Mercator()
    ax = plt.axes(projection=projection)

    # Cartopy transparency
    ax.outline_patch.set_visible(False)
    ax.background_patch.set_visible(False)
    ax.background_patch.set_alpha(0)

    # Create an animation
    rcParams['mathtext.default'] = 'regular'

    # Make a segmented colour map for the rain.
    rain_cm = plt.get_cmap('Blues', 10)

    # Plots the requested time from the model output
    too_old = False
    if fname.exists():
        file_age = (datetime.now() - datetime.fromtimestamp(fname.stat().st_mtime)).total_seconds()
        if file_age > 24 * 60 * 60:
            too_old = True
    if not fname.exists() or overwrite or too_old:
        # We later check that the files we have been asked to make actually exist, so best to remove them before we make
        # a new one.
        if fname.exists():
            logger.debug(f'Removing {fname}')
            fname.unlink()
        else:
            logger.debug(f'No existing {fname} found')

        logger.debug(f'Creating {fname}')
        ax.clear()
        ax.axis('off')
        ax.contour(x,
                   y,
                   pressure,
                   levels=np.arange(0, 1000, 5),
                   colors=['black'],
                   nchunk=5,
                   transform=ccrs.PlateCarree(),
                   zorder=50)
        cf = ax.contourf(x,
                   y,
                   rain,
                   transform=ccrs.PlateCarree(),
                   linestyles=None,
                   alpha=0.75,
                   zorder=150,
                   cmap=rain_cm,
                   norm=LogNorm())
        cf.set_clim(vmax=2)
        # Convert Kelvin to Celsius here
        add_cities(ax, locations['cities'], x, y, temperature - 273.15)
        # ax.coastlines(zorder=200, color='w', linewidth=1)
        ax.set_extent((x.min(), x.max(), y.min(), y.max()), crs=ccrs.PlateCarree())
        fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=96, transparent=True)
    else:
        logger.debug(f'{fname} already exists and overwrite is {overwrite}')

    plt.close()


def make_ocean_frame(fname, x, y, temperature, salinity, u, v, overwrite=False):
    """
    Plot an ocean model output frame to file `fname' with the given:

        - x, y: coordinate arrays (spherical)
        - temperature, salinity, u, v: data arrays (same size as x, y)

    Optionally pass `overwrite' as True to overwrite existing files, otherwise files are only overwritten if they're
    over 24 hours old. Defaults to False.

    """

    fig = plt.figure(figsize=(12, 10))
    projection = ccrs.Mercator()
    ax = plt.axes(projection=projection)

    # Cartopy transparency
    ax.outline_patch.set_visible(False)
    ax.background_patch.set_visible(False)
    ax.background_patch.set_alpha(0)

    # Create an animation
    rcParams['mathtext.default'] = 'regular'

    # Plots the requested time from the model output
    too_old = False
    file_age = (datetime.now() - datetime.fromtimestamp(fname.stat().st_mtime)).total_seconds()
    if fname.exists() and file_age > 24 * 60 * 60:
        too_old = True
    if not fname.exists() or overwrite or too_old:
        logger.debug(f'Creating {fname}')
        ax.clear()
        ax.axis('off')
        ax.pcolormesh(x,
                      y,
                      np.squeeze(temperature),
                      cmap=cm.thermal,
                      vmin=5,
                      vmax=20,
                      transform=ccrs.PlateCarree(),
                      zorder=40)
        ax.contour(x,
                  y,
                  np.squeeze(salinity),
                  levels=np.arange(10, 35, 0.25),
                  colors=['white'],
                  nchunk=5,
                  transform=ccrs.PlateCarree(),
                  zorder=50)
        ax.quiver(x[::10, ::10],
                  y[::10, ::10],
                  u[::10, ::10],
                  v[::10, ::10],
                  color='0.6',
                  scale=50,
                  transform=ccrs.PlateCarree(),
                  zorder=60)
        ax.set_extent((x.min(), x.max(), y.min(), y.max()), crs=ccrs.PlateCarree())
        fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=96, transparent=True)
    else:
        logger.debug(f'{fname} already exists and overwrite is {overwrite}')

    plt.close()


def make_video(meta, source='pml', map_type='atmosphere', overwrite=False, serial=False):
    """
    For the given source model and map type, plot the frames for the map.

    Set `overwrite' to True to overwrite existing files. Defaults to only overwriting if the existing files are over 24
    hours old.

    Set `serial` to True to create the frames in serial. Defaults to spinning up a multiprocessing pool for creating
    the frames.

    """

    logger.info(f'Fetching weather forecast from {source}')
    logger.debug(f'Source URL is {meta["url"]}')

    if source == 'gfs' and map_type == 'ocean':
        raise ValueError('Ocean maps not yet supported for GFS source data')

    logger.info(f'Plotting {map_type} from {source}')
    ncvars, dims = get_ncvars(source, map_type)

    ds = meta['ds']

    skip_offset = 0
    if source == 'pml':
        if map_type == 'atmosphere':
            skip_offset = 1  # skip the initial condition
        elif map_type == 'ocean':
            skip_offset = 0  # not sure what's hindcast in this

    logger.debug('Fetching time')
    time = num2date(ds.variables[ncvars['time']][:], ds.variables[ncvars['time']].units)[skip_offset:]

    # We can now check whether files exist and if we're overwriting
    logger.debug('Check for existing frames on disk')
    fnames = []
    missing_frames = []
    for i in range(ds.dimensions[dims['time']].size - skip_offset - 1):
        fname = Path('static', 'dynamic', 'frames', f'{source}_{map_type}_frame_{i + 1:02d}.png')
        fname.parent.mkdir(parents=True, exist_ok=True)
        too_old = False
        if fname.exists():
            file_age = (datetime.now() - datetime.fromtimestamp(fname.stat().st_mtime)).total_seconds()
            if file_age > 24 * 60 * 60:
                too_old = True
        if not fname.exists() or overwrite or too_old:
            missing_frames.append(True)
        else:
            missing_frames.append(False)
        fnames.append(fname)

    # If we don't have any missing frames and we're not overwriting, we can just skip all the expensive network stuff
    # and use what we have on disk.
    if any(missing_frames):
        logger.info(f'Creating {sum(missing_frames)} new frames')
    else:
        logger.info('No missing frames and we are not overwriting; skip out early')
        return

    if map_type == 'atmosphere':
        logger.debug('Fetching atmosphere variables')

        logger.debug('Fetching rain')
        if source == 'pml':
            # Convert from cumulative rainfall to instantaneous in m/s. Look away efficiency fans. It's 17:51 and I need
            # to go home, so this'll have to do.
            logger.info('De-accumulating the rain data')
            rain = ds.variables[ncvars['rain']]
            temp = np.zeros(rain[skip_offset:].shape)
            for gi in range(1, rain.shape[0] - skip_offset):
                temp[gi] = np.diff(rain[gi - 1:gi + 1, ...], axis=0) / 1000 / (time[-1] - time[-2]).total_seconds()
            # The initial conditions (GFS). We don't want that here, so skip this for now.
            # temp[0] = rain[0] / 1000 / (time[-1] - time[-2]).total_seconds()
            rain = temp * 3600 * 24  # into m/d
            del temp
            # Remove tiiiiny rainfall values
            rain = np.ma.masked_where(rain < 0.001, rain)
        else:
            rain = ds.variables[ncvars['rain']]

        # Convert pressure to millibars.
        logger.debug('Fetching pressure')
        if source == 'pml':
            # For WRF, we do some jiggery-pokery to get surface pressure.
            surface_pressure = ds.variables['PSFC'][1:]
            base_pressure = ds.variables['PB'][1:, 0]
            pressure = (surface_pressure - base_pressure) / 100
        else:
            pressure = ds.variables[ncvars['surface_pressure']][:] / 100

        logger.debug('Fetching temperature')
        if source == 'pml':
            temperature = ds.variables[ncvars['temperature']][1:]
        else:
            temperature = ds.variables[ncvars['temperature']]

        with Path('cities.yaml').open('r', encoding='utf8') as f:
            locations = safe_load(f)
    elif map_type == 'ocean':
        logger.debug('Fetching ocean variables')
        u = ds.variables[ncvars['u']]
        v = ds.variables[ncvars['v']]
        temperature = ds.variables[ncvars['temperature']]
        salinity = ds.variables[ncvars['salinity']]

    # Save the animation frames to disk.
    logger.debug('Animating frames')
    if serial:
        for i in range(ds.dimensions[dims['time']].size - skip_offset - 1):
            logger.debug(f'Fetch data for time index {i}')
            si = skip_offset + i
            if map_type == 'atmosphere':
                make_atmosphere_frame(fnames[i],
                                      meta['x'],
                                      meta['y'],
                                      pressure[si],
                                      rain[si],
                                      temperature[si],
                                      locations,
                                      overwrite)
            elif map_type == 'ocean':
                # Surface fields only
                make_ocean_frame(fnames[i],
                meta['x'],
                meta['y'],
                temperature[si, 0],
                salinity[si, 0],
                u[si, 0],
                v[si, 0],
                overwrite)
    else:
        with multiprocessing.Pool() as pool:
            args = []
            for i in range(ds.dimensions[dims['time']].size - skip_offset - 1):
                logger.debug(f'Fetch data for time index {i}')
                si = skip_offset + i
                if map_type == 'atmosphere':
                    args.append((fnames[i],
                                meta['x'],
                                meta['y'],
                                pressure[si],
                                rain[si],
                                temperature[si],
                                locations,
                                overwrite))
                    fn = make_atmosphere_frame
                elif map_type == 'ocean':
                    args.append((fnames[i],
                                meta['x'],
                                meta['y'],
                                temperature[si, 0],
                                salinity[si, 0],
                                u[si, 0],
                                v[si, 0],
                                overwrite))
                    fn = make_ocean_frame
            try:
                pool.starmap(fn, args)
            except ValueError:
                pass

    # Check for all the frames and if they're there, call that a success
    made_frames = all([i.exists() for i in fnames])

    return made_frames


def wind_chill(temp, wind_speed):
    """
    Compute wind chill based on temperature and wind speed.

    Taken from https://en.wikipedia.org/wiki/Wind_chill#North_American_and_United_Kingdom_wind_chill_index.

    """

    # Wind chill is only calculated where temperatures are below 10 Celsius and wind speeds are above 4.8 kph. Mask
    # off values which fall outside those ranges.
    mask_temp = temp <= 10
    mask_wind = (wind_speed * 60 * 60 / 1000) > 1 + (1/3)
    mask = np.bitwise_and(mask_temp, mask_wind)

    temp = np.ma.masked_array(temp, mask=~mask)
    wind_speed = np.ma.masked_array(wind_speed, mask=~mask) * 60 * 60 / 1000

    chill = 13.12 + (0.6215 * temp) - (11.37 * wind_speed**0.16) + (0.3965 * temp * wind_speed**0.16)

    # Replace masked values with the original ones
    chill[np.argwhere(~mask)] = temp[np.argwhere(~mask)]

    # Return the data values only
    return chill.data
