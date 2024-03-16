import csv
import matplotlib.pyplot as plt
import random
import json
from geopy.distance import geodesic
import math


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

i = 0
for row in rows:
    # if i > 10000: break
    i+=1
    lat = float(row['lat'])
    lon = float(row['lng'])
    country = row['country']
    if country != "Japan": continue

    # If the country is not in the dictionaries, add it
    if country not in latitudes:
        latitudes[country] = []
        longitudes[country] = []
        colors[country] = get_random_color()

    latitudes[country].append(lat)
    longitudes[country].append(lon)
    if country in countries: countries[country] += 1 
    else: countries[country] = 1


def getScore(x, y):
    total = 0
    for i in range(len(latitudes["Japan"])):
        location1 = (x, y)  
        location2 = (latitudes["Japan"][i], longitudes["Japan"][i]) 
        
        distance = geodesic(location1, location2).kilometers
        score = math.ceil(5000*pow(2, -0.00078*distance))
        total += score
    total/=len(latitudes["Japan"])
    return total


import matplotlib.cm as cm

def hill_climb(start_coordinate, step_size, max_iterations, ax):
    current_coordinate = start_coordinate
    current_score = getScore(current_coordinate[0], current_coordinate[1])

    # Create a color map from black to red
    color_map = cm.get_cmap('Reds')

    # Normalize the score to the range [0, 1] for the color map
    normalized_score = current_score / 5000

    # Plot the start coordinate with the color corresponding to the score
    ax.plot(current_coordinate[1], current_coordinate[0], 'o', color=color_map(normalized_score))

    for iteration in range(max_iterations):
        # Generate a new candidate coordinate
        candidate_coordinate = (current_coordinate[0] + random.uniform(-1, 1) * step_size,
                                current_coordinate[1] + random.uniform(-1, 1) * step_size)

        # Calculate the score of the new coordinate
        candidate_score = getScore(candidate_coordinate[0], candidate_coordinate[1])

        # If the new coordinate's score is better, update the current coordinate and score
        if candidate_score > current_score:
            # Draw a line from the current coordinate to the candidate coordinate
            ax.plot([current_coordinate[1], candidate_coordinate[1]], [current_coordinate[0], candidate_coordinate[0]], 'k-')

            current_coordinate = candidate_coordinate
            current_score = candidate_score

            # Normalize the score to the range [0, 1] for the color map
            normalized_score = max(0, current_score-2000) / 2000

            # Plot the new coordinate with the color corresponding to the score
            ax.plot(current_coordinate[1], current_coordinate[0], 'o', color=color_map(normalized_score))

        print(f"Iteration {iteration}: coordinate = {current_coordinate}, score = {current_score}")

    return current_coordinate

print("Osaka:",getScore(34.680111, 135.410160))
print("Nagoya:",getScore(35.044324, 136.859266))
print("Tokyo:",getScore(35.811150, 139.487628))
print("Scriptet (36.02473831974561, 137.09922493923457):",getScore(34.680111, 135.410160))

# Create a subplot for the hill climbing algorithm
fig, ax = plt.subplots()

current_coordinate = hill_climb(start_coordinate=(38.787558, 136.438214), step_size=5, max_iterations=100, ax=ax)
current_coordinate = hill_climb(start_coordinate=current_coordinate, step_size=3, max_iterations=100, ax=ax)
current_coordinate = hill_climb(start_coordinate=current_coordinate, step_size=1, max_iterations=100, ax=ax)
current_coordinate = hill_climb(start_coordinate=current_coordinate, step_size=0.5, max_iterations=100, ax=ax)

sorted_countries = {k: v for k, v in sorted(countries.items(), key=lambda item: -item[1])}
for k, v in sorted_countries.items():
    print(k,"has", v,"locations")

percentage_to_display = 1


# Plot the points for each country
for country in latitudes:
    num_points = int(len(latitudes[country]) * percentage_to_display)
    ax.scatter(longitudes[country][:num_points], latitudes[country][:num_points], s=1, color=colors[country])


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
