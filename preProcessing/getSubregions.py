import geopandas as gpd
from shapely.geometry import Point
import csv
import random

brazil = gpd.read_file(r'Data\bra_adm_ibge_2020_shp\bra_admbnda_adm1_ibge_2020.shp')

def get_region_from_coordinates(lat, long):
    coords = [Point(long, lat)]  
    gdf = gpd.GeoDataFrame(geometry=coords)

    result = gpd.sjoin(gdf, brazil, predicate='within')
    try:
        state = result['ADM1_PT'].values[0]
        return state
    except IndexError:
        print("Coordinates are not in Brazil")
        print(lat, long)
        return "-1"

# Load the CSV file
with open(r'Data\archive\images.csv', 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Create dictionaries for latitudes, longitudes, dates, and states
latitudes = {}
longitudes = {}
dates = {}
states = {}

i = 0
for row in rows:
    i+=1
    lat = float(row['lat'])
    lon = float(row['lng'])
    date = row['date_taken']
    country = row['country']
    if country != "Brazil": continue

    state = get_region_from_coordinates(lat, lon)
    if(state == "-1"): continue

    if state not in latitudes:
        latitudes[state] = []
        longitudes[state] = []
        dates[state] = []
        states[state] = []

    latitudes[state].append(lat)
    longitudes[state].append(lon)
    dates[state].append(date)
    states[state].append(row['id'])

# Write the data to a new CSV file
with open(r'Data\archive\brazil.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "lat", "lng", "date_taken", "state"])  # Write the header

    # Write the data for each state
    for state in states:
        for id, lat, lng, date in zip(states[state], latitudes[state], longitudes[state], dates[state]):
            writer.writerow([id, lat, lng, date, state])
