from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
 
# Creating dataset
data = pd.read_csv('/data/transformer-metarl/garage/examples/torch/embeddings/representation.csv').values
metadata = pd.read_csv('/data/transformer-metarl/garage/examples/torch/embeddings/metadata.csv').values[:, 1]
z = data[:, 0]
x = data[:, 1]
y = data[:, 2]
 
colors = cm.Dark2(np.linspace(0, 0.8, 7))

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

for i in range(7):
    # Creating plot
    ax.scatter3D(x[400*i: 400*(i + 1)], y[400*i: 400*(i + 1)], z[400*i: 400*(i + 1)], color=colors[i], label=metadata[400*i])

plt.legend(title='Velocity')
plt.title("HalfCheetahVel - Working Memories")
 
# show plot
plt.show()