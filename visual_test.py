import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Create a new figure and Axes3D object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add some data to the plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
z = [3, 6, 9, 12, 15]
ax.scatter(x, y, z)

# Show the plot
plt.show()
