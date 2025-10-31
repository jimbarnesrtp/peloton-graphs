# peloton-graphs
# Install deps
pip install requests matplotlib python-dateutil selenium webdriver-manager

Must have chrome installed on your machine

# Today in New York, single graph per ride, cleanup first
python peloton_date_graphs.py --username you@example.com --tz "America/New_York" --cleanup

# Specific date + two graphs per ride (power/zones and HR/Cad/Res), cleanup first
python peloton_date_graphs.py --username you@example.com --tz "America/Los_Angeles" --date 2025-10-20 --two-graphs --cleanup

# Show all IANA timezones (Python 3.9+)
python peloton_date_graphs.py --list-timezones

# Help (also prints automatically if you run with no args)
python peloton_date_graphs.py --help

#use a stacked graph
python peloton_date_graphs.py --username you@example.com --tz "America/New_York" --cleanup --stacked

#peloton_cookie.py will launch a chrome browser, you will be prompted to login and then it will go call the API
python peloton_cookie.py --tz "America/New_York"

