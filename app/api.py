import logging

import flask
import numpy as np
from netCDF4 import num2date

from utils import get_current_forecast_metadata, wind_chill

api = flask.Blueprint('api', __name__)
logger = logging.getLogger(__name__)


@api.route('/timeseries')
def timeseries():
    # Find the closest location to that which has been passed in
    x = float(flask.request.args.get('lon'))
    y = float(flask.request.args.get('lat'))

    meta = get_current_forecast_metadata()

    values = {'status': 200}

    # Checking for values inside the bounding box is a bit tricky as it's possible to click under the arched bottom
    # of the map, for example, and satisfy the criteria below and yet still not have any valid data.
    if meta['west'].min() < x < meta['east'].max() and meta['south'].min() < y < meta['north'].max():
        logger.debug(f"{x}/{y} inside the model domain ({meta['west'].min()}/{meta['east']}/{meta['south']}/{meta['north']})")
        ds = meta['ds']
        pos_row, pos_col = np.unravel_index(np.argmin(np.hypot(meta['x'] - x, meta['y'] - y)), meta['x'].shape)

        logger.debug('Fetching time')
        time = [i.strftime('%Y-%m-%d %H:%M:%S') for i in num2date(ds['XTIME'], ds.variables['XTIME'].units)]
        logger.debug('Fetching temperature')
        temp = ds['T2'][:, pos_row, pos_col] - 273.15  # to Celsius

        logger.debug('Fetching pressure')
        surface_pressure = ds.variables['PSFC'][:, pos_row, pos_col]
        base_pressure = ds.variables['PB'][:, 0, pos_row, pos_col]
        pressure = (surface_pressure - base_pressure) / 100

        logger.debug('Fetching wind components')
        u = ds.variables['U10'][:, pos_row, pos_col]
        v = ds.variables['V10'][:, pos_row, pos_col]
        wind_speed = np.hypot(u, v)
        wind_direction = np.rad2deg(np.arctan2(v, u))

        temp_wind_chill = wind_chill(temp, wind_speed)
        logger.debug(', '.join(temp.astype(str)))
        logger.debug(', '.join(wind_speed.astype(str)))

        values.update({'requested_lon': x,
                       'requested_lat': y,
                       'grid_lon': meta['x'][pos_row, pos_col].astype(float),
                       'grid_lat': meta['y'][pos_row, pos_col].astype(float),
                       'utc_time': time,
                       'temperature_celsius': temp.tolist(),
                       'temperature_celsius_wind_chill': temp_wind_chill.tolist(),
                       'surface_pressure_mb': pressure.tolist(),
                       'wind_speed_ms-1': wind_speed.tolist(),
                       'wind_direction_degN': wind_direction.tolist()})
    else:
        logger.debug(f"{x}/{y} outside the model domain ({meta['west'].min()}/{meta['east']}/{meta['south']}/{meta['north']})")
        values['status'] = 500

    return flask.jsonify(values)
