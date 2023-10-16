import csv
import json
from shapely.geometry import Point, Polygon
import time

country_polygons = []
start_time = time.time()


# Load the JSON file
def loadJson():
    with open(r'Data\world-administrative-boundaries.json', 'r') as f:
        data = json.load(f)
    return data


def addPolygons(data):
    # Loop over each dictionary in the list
    for country in data:
        # before = len(country_polygons)
        # Extract the coordinates
        coordinates = country['geo_shape']['geometry']['coordinates']

        # Create polygons for each countr
        if isinstance(coordinates[0][0][0], list):
            for polygons in coordinates:
                for border in polygons:
                    polygon = Polygon(border)
                    country_polygons.append([0, country['name'], polygon])
        else:
            for border in coordinates:
                polygon = Polygon(border)
                country_polygons.append([0, country['name'], polygon])
        # print(country["name"], len(country_polygons)-before)


def identifyCountry(row):
        lat = float(row['lat'])
        lon = float(row['lng'])
        point = Point(lon, lat)

        closestCountry = [country_polygons[0][2].exterior.distance(point), 0]
        
        for j in range(len(country_polygons)):
            freq, name, polygon = country_polygons[j]
            if polygon.contains(point):  # Adjust this threshold as needed
                if name == "South Africa":
                    closestCountry = [-1, j]
                    continue
                row['country'] = name  # Add the country information to the row
                country_polygons[j][0] += 1
                return
            closestCountry = min(closestCountry, [polygon.exterior.distance(point), j])
        country_polygons[closestCountry[1]][0] += 1
        row["country"] = country_polygons[closestCountry[1]][1]
               

def findCountries():   
    global start_time 
    # Load the CSV file
    with open(r'Data\archive\images.csv', 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    for i, row in enumerate(rows):
        identifyCountry(row)
        # Print progress every second
        if time.time() - start_time > 1:
            print(f"Progress: {((i+1)/len(rows))*100}% locations processed")
            start_time = time.time()
            country_polygons.sort(key=lambda x:-x[0])
        

    # Write the updated rows back to the CSV file
    with open(r'Data\archive\images.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames + ['country'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    addPolygons(loadJson())
    print(len(country_polygons))
    findCountries()


if __name__ == "__main__":
    main()

