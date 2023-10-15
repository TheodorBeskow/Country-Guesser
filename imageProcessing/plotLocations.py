import csv
import matplotlib.pyplot as plt

percentage_to_display = 1  

with open(r'Data\archive\images.csv', 'r') as file:
    reader = csv.DictReader(file)
    latitudes = []
    longitudes = []
    for row in reader:
        lat = float(row['lat'])
        lon = float(row['lng'])
        latitudes.append(lat)
        longitudes.append(lon)

num_points = int(len(latitudes) * percentage_to_display)

plt.scatter(longitudes[:num_points], latitudes[:num_points], s=1)
plt.title('Plot of Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
