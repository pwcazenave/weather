import logging

import flask
import numpy as np
from netCDF4 import num2date

from utils import get_current_forecast_metadata

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
    if meta['west'] < x < meta['east'].max() or meta['south'].min() < y < meta['north'].max():
        logger.debug(f"{x}/{y} inside the model domain ({meta['west'].min()}/{meta['east']}/{meta['south']}/{meta['north']})")
        ds = meta['ds']
        pos_row, pos_col = np.unravel_index(np.argmin(np.hypot(meta['x'] - x, meta['y'] - y)), meta['x'].shape)

        time = [i.strftime('%Y-%m-%d %H:%M:%S') for i in num2date(ds['XTIME'], ds.variables['XTIME'].units)]
        temp = [float(i) for i in ds['T2'][:, pos_row, pos_col] - 273.15]  # to Celsius

        surface_pressure = ds.variables['PSFC'][:, pos_row, pos_col]
        base_pressure = ds.variables['PB'][:, 0, pos_row, pos_col]
        pressure = [float(i) for i in (surface_pressure - base_pressure) / 100]

        values.update({'requested_lon': x,
                       'requested_lat': y,
                       'grid_lon': meta['x'][pos_row, pos_col].astype(float),
                       'grid_lat': meta['y'][pos_row, pos_col].astype(float),
                       'utc_time': time,
                       'temperature_celsius': temp,
                       'surface_pressure_mb': pressure})
    else:
        logger.debug(f"{x}/{y} outside the model domain ({meta['west'].min()}/{meta['east']}/{meta['south']}/{meta['north']})")
        values['status'] = 500

    return flask.jsonify(values)
