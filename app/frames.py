#!/opt/weather/venv/bin/python3
#
# Script to generate the frames for the weather map forecasts.

import logging
import sys

# Import the necessary bits and pieces from the utils
import utils

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)
logger.warning('test')

source, map_type = sys.argv[1:]
logger.warning(f'Creating new frames from {source} for {map_type}')
meta = utils.get_current_forecast_metadata(source, map_type)
utils.make_video(meta, source=source, map_type=map_type, overwrite=False, serial=False)
