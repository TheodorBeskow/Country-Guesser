import matplotlib.pyplot as plt
import json

with open(r'Data\world-administrative-boundaries.json', 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots()

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
    else: print("Error with country:", country["name"])

plt.show()
