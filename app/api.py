import logging

import flask
import numpy as np
from netCDF4 import num2date

from utils import get_current_forecast_metadata, wind_chill

api = flask.Blueprint('api', __name__)
logger = logging.getLogger(__name__)


def get_ncvars(source, map_type):
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

    return ncvars


@api.route('/timeseries/<source>/<map_type>')
def timeseries(source, map_type):
    ncvars = get_ncvars(source, map_type)

    # Find the closest location to that which has been passed in
    x = float(flask.request.args['lon'])
    y = float(flask.request.args['lat'])

    meta = get_current_forecast_metadata(source, map_type)

    values = {'status': 200}

    # Checking for values inside the bounding box is a bit tricky as it's possible to click under the arched bottom
    # of the map, for example, and satisfy the criteria below and yet still not have any valid data.
    if meta['west'].min() < x < meta['east'].max() and meta['south'].min() < y < meta['north'].max():
        logger.debug(f"{x}/{y} inside the model domain ({meta['west'].min()}/{meta['east']}/{meta['south']}/{meta['north']})")
        ds = meta['ds']
        pos_row, pos_col = np.unravel_index(np.argmin(np.hypot(meta['x'] - x, meta['y'] - y)), meta['x'].shape)

        logger.debug('Fetching time')
        time = [i.strftime('%Y-%m-%d %H:%M:%S') for i in num2date(ds.variables[ncvars['time']], ds.variables[ncvars['time']].units)]

        if map_type == 'atmosphere':
            logger.debug('Fetching temperature')
            temp = ds.variables[ncvars['temperature']][:, pos_row, pos_col] - 273.15  # to Celsius
            logger.debug('Fetching pressure')
            if source == 'pml':
                surface_pressure = ds.variables[ncvars['surface_pressure']][:, pos_row, pos_col]
                base_pressure = ds.variables[ncvars['base_pressure']][:, 0, pos_row, pos_col]
                pressure = (surface_pressure - base_pressure) / 100
            elif source == 'gfs':
                pressure = ds.variables[ncvars['surface_pressure']][:, pos_row, pos_col] / 100

            logger.debug('Fetching wind components')
            u = ds.variables[ncvars['u']][:, pos_row, pos_col]
            v = ds.variables[ncvars['v']][:, pos_row, pos_col]
            wind_speed = np.hypot(u, v)
            wind_direction = np.rad2deg(np.arctan2(v, u))

            temp_wind_chill = wind_chill(temp, wind_speed)
        elif map_type == 'ocean':
            logger.debug('Fetching temperature')
            temp = ds.variables[ncvars['temperature']][:, 0, pos_row, pos_col]
            logger.debug('Fetching salinity')
            salinity = ds.variables[ncvars['salinity']][:, 0, pos_row, pos_col]
            logger.debug('Fetching velocity')
            u = ds.variables[ncvars['u']][:, 0, pos_row, pos_col]
            v = ds.variables[ncvars['v']][:, 0, pos_row, pos_col]

        # Common variables
        values.update({'requested_lon': x,
                       'requested_lat': y,
                       'grid_lon': meta['x'][pos_row, pos_col].astype(float),
                       'grid_lat': meta['y'][pos_row, pos_col].astype(float),
                       'utc_time': time,
                       'temperature_celsius': temp.tolist()})

        if map_type == 'atmosphere':
            values.update({'temperature_celsius_wind_chill': temp_wind_chill.tolist(),
                           'surface_pressure_mb': pressure.tolist(),
                           'wind_speed_ms-1': wind_speed.tolist(),
                           'wind_direction_degN': wind_direction.tolist()})
        elif map_type == 'ocean':
            values.update({'salinity_psu': salinity.tolist(),
                           'u_m_per_s': u.tolist(),
                           'v_m_per_s': v.tolist()})
    else:
        logger.debug(f"{x}/{y} outside the model domain ({meta['west'].min()}/{meta['east']}/{meta['south']}/{meta['north']})")
        values['status'] = 500

    return flask.jsonify(values)
