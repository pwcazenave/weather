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
from datetime import datetime
from pathlib import Path

import flask
import flask_wtf
from flask_apscheduler import APScheduler

from api import api
import utils

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
logger.info('Starting scheduler')
scheduler.start()

# Configure CSRF protection.
csrf = flask_wtf.csrf.CSRFProtect(app)
app.config['SECRET_KEY'] = os.urandom(32)
csrf.init_app(app)

# Remove unnecessary whitespace in the rendered HTML
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True


@app.route('/')
def slash():
    return flask.redirect(flask.url_for('create_map', map_type='atmosphere'))


@app.route('/map/<map_type>')
def create_map(map_type):
    source = 'pml'
    if source == 'pml':
        if map_type == 'ocean':
            num_frames = 73
        elif map_type == 'atmosphere':
            num_frames = 32
    elif source == 'gfs':
        num_frames = 10  # ?

    meta = utils.get_current_forecast_metadata(source, map_type)
    kwargs = {'west': meta['west'],
              'east': meta['east'],
              'south': meta['south'],
              'north': meta['north'],
              'api_key': api_key,
              'num_frames': num_frames,
              'map_type': map_type,
              'source': source}
    # Make sure we have the frames for today
    utils.make_video(meta, source=source, map_type=map_type, overwrite=False, serial=False)

    return flask.render_template('map.html', **kwargs)


@app.route('/contact')
def contact():
    return flask.render_template('contact.html')


@app.route('/about')
def about():
    return flask.render_template('about.html')


@scheduler.task('cron', id='make_video', day='*', hour=2, minute=30)
def today_video():
    # Create frames for the most recent model runs
    meta = utils.get_current_forecast_metadata(source='pml', map_type='ocean')
    utils.make_video(meta, overwrite=True)
    meta = utils.get_current_forecast_metadata(source='pml', map_type='atmosphere')
    utils.make_video(meta, overwrite=True)


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
            extra_files=['./app/static/favicon.ico',
                         './app/static/css/style.css'])


if __name__ == '__main__':

    logger.info('Starting main')
    main()
