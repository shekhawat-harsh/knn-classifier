import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from  matplotlib import style

def distance(p, q):
    [a, b] = np.array(p) - np.array(q)
    distance = np.sqrt(a**2 + b**2)
    return distance

class KNNAlgorithm:
    def __init__(self, points, newPoint):
        self.points = points
        self.newPoint = newPoint

    def predict(self):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distances.append([category, distance(point, self.newPoint)])
        
        # Sort distances by the distance value
        sorted_distances = sorted(distances, key=lambda x: x[1])

        # Get the categories of the k nearest neighbors (k=2 in this case)
        nearest_categories = [sorted_distances[i][0] for i in range(2)]

        # Return the most common category
        return Counter(nearest_categories).most_common(1)[0][0]

points = {"blue": [[2, 4], [2, 2], [4, 5]], "red": [[3, 4], [5, 6], [9, 0]]}
new_point = [4, 4]

result = KNNAlgorithm(points=points, newPoint=new_point).predict()

print(result)



# visualsie

for point in points["blue"]:

    plt.scatter( x = point[0], y =  point[1] , color = "blue")
    plt.plot([new_point[0] , point[0]],[new_point[1] , point[1]] , color = 'blue')
for point in points["red"]:

    plt.scatter(point[0] , point[1] , color = 'red' )
    plt.plot([new_point[0] , point[0]],[new_point[1] , point[1]] , color = 'red')

plt.scatter(new_point[0] , new_point[1] , color = 'black')

plt.show()