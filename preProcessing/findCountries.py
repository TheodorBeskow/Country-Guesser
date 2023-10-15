import csv
import json
from shapely.geometry import Point, Polygon
from rtree import index
import time

# Load the GeoJSON file
with open(r'Data\world-administrative-boundaries.json', 'r') as f:
    data = json.load(f)

# Create an R-tree index
idx = index.Index()

# Create a dictionary to store each country's polygons and colors
country_polygons = {}

# Loop over each dictionary in the list
for i, country in enumerate(data):
    # Extract the coordinates
    coordinates = country['geo_shape']['geometry']['coordinates']

    # Create polygons for each country and add them to the R-tree index
    if len(coordinates) > 0 and not isinstance(coordinates[0][0][0], list):
        for border in coordinates:
            polygon = Polygon(border)
            idx.insert(i, polygon.bounds)
            country_polygons[i] = (country['name'], polygon)

# Load the CSV file
with open(r'Data\archive\images.csv', 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

start_time = time.time()

# Add the country information to each row in the CSV file
for i, row in enumerate(rows):
    lat = float(row['lat'])
    lon = float(row['lng'])
    point = Point(lon, lat)
    
    # Find the nearest polygon using the R-tree index
    for j in idx.nearest((point.x, point.y, point.x, point.y), 1):
        name, polygon = country_polygons[j]
        row['country'] = name  # Add the country information to the row
        break

    # Print progress every 3 seconds
    if time.time() - start_time > 3:
        print(f"Progress: {((i+1)/len(rows))*100}% locations processed")
        start_time = time.time()

# Write the updated rows back to the CSV file
with open(r'Data\archive\images.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=reader.fieldnames + ['country'])
    writer.writeheader()
    writer.writerows(rows)

