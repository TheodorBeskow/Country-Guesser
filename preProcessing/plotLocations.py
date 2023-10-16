import csv
import matplotlib.pyplot as plt
import random
import json

# Define a function to generate random colors
def get_random_color():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())

# Load the CSV file
with open(r'Data\archive\images.csv', 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Create dictionaries for latitudes, longitudes, and colors
latitudes = {}
longitudes = {}
colors = {}
countries = {}

for row in rows:
    lat = float(row['lat'])
    lon = float(row['lng'])
    country = row['country']

    # If the country is not in the dictionaries, add it
    if country not in latitudes:
        latitudes[country] = []
        longitudes[country] = []
        colors[country] = get_random_color()

    latitudes[country].append(lat)
    longitudes[country].append(lon)
    if country in countries: countries[country] += 1 
    else: countries[country] = 1

sorted_countries = {k: v for k, v in sorted(countries.items(), key=lambda item: -item[1])}
for k, v in sorted_countries.items():
    print(k,"has", v,"locations")

percentage_to_display = 1

fig, ax = plt.subplots()

# Plot the points for each country
for country in latitudes:
    num_points = int(len(latitudes[country]) * percentage_to_display)
    ax.scatter(longitudes[country][:num_points], latitudes[country][:num_points], s=1, color=colors[country])


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
