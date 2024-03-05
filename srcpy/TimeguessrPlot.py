import json
import matplotlib.pyplot as plt
from collections import Counter

# Load the data from the JSON file
with open('Data/TimeGuessr/data.json') as f:
    data = json.load(f)

# Extract the coordinates
coordinates = [item['Location'] for item in data]

# Separate the latitudes and longitudes
latitudes = [coord['lat'] for coord in coordinates]
longitudes = [coord['lng'] for coord in coordinates]

# Create a new figure and axes
fig, ax = plt.subplots()

# Plot the coordinates
ax.scatter(longitudes, latitudes, s=30, alpha=0.2)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Plot of Coordinates')

# Load the country borders data
with open(r'Data\world-administrative-boundaries.json', 'r') as f:
    data = json.load(f)

# Plot the country borders
for country in data:
    coordinates = country['geo_shape']['geometry']['coordinates']

    if len(coordinates) > 0 and not isinstance(coordinates[0][0][0], list):
        for border in coordinates:
            x, y = zip(*border)
            ax.plot(x, y, color='black')

    elif len(coordinates) > 0 and isinstance(coordinates[0][0][0], list):
        for landmass in coordinates:
            for border in landmass:
                x, y = zip(*border)
                ax.plot(x, y, color='black')
    else: 
        print("Error with country:", country["name"])

plt.show()
