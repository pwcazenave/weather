# Weather

Simple webapp to display model outputs from a THREDDS server over a https://here.com map.

# Installation

Needs python3. Once you have it:

```bash
git clone git@github.com:pwcazenave/weather.git
cd weather
python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt
```j

You need to register with Here maps and [get an API key](https://developer.here.com/documentation/maps/3.1.29.0/dev_guide/topics/quick-start.html). Once you have that key, run the application as:

```bash
API_KEY=theheremapsapitokenstring DEBUG=1 python3 app/main.py
```

That should serve the application on http://localhost:8000. If you don't have the API key, most stuff should still work except there won't be a map background.

# Demo

I've put up a demo version at https://weather.cazenave.co.uk.

