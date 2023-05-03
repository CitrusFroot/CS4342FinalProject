import numpy as np
import matplotlib.pyplot as plt

# Load the data
dataset = np.loadtxt('Kannada-MNIST/Dig-MNIST.csv', delimiter=',', skiprows=1)
labels = dataset[:, 0]  # first column = y
dataset = dataset[:, 1:]  # shape = (#imgs, pixels)

# Reshape the data
dataset = np.reshape(dataset, (-1, 28, 28))

# Define the labels
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Plot some examples
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(dataset[i], cmap='gray')
    ax.set_title(label_names[int(labels[i])])
    ax.axis('off')
plt.show()
