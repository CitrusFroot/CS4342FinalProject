import numpy as np
import pandas as pd
import tensorflow as tf

ADDRESS = 'Put the address to all the images here'
BATCHSIZE = 0
EPOCHCOUNT = 0
IMAGESIZE = [720,1280]
VALIDATIONSPLIT = 0.3

#Returns the image names, labels, and baseline accuracy labels
def getBaseline():
    data = pd.read_csv(r'fathomnet-out-of-sample-detection\multilabel_classification\train.csv')
    categories = pd.DataFrame(data, columns = ['categories'])
    images = pd.DataFrame(data, columns = ['id'])

    categories = categories.to_numpy(categories.reset_index(drop = True, inplace = True)) #removes unnecessary info
    images = images.to_numpy(images.reset_index(drop = True, inplace = True)) #same as above

    categories = np.ndarray.flatten(categories) #flatten to get 1d array rather than 2d; note that they are strings, not ints
    images = np.ndarray.flatten(images) #same as above

    ybase = [] #TODO

    return(images, categories, ybase)

#loads in training and testing data
#returns: the training and testing dataset, as well as the ybase
def load_data():
    images, labels, ybase = getBaseline()

    images = normalize(images) #we call this to load the images AS images, not filenames, and to normalize them

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(BATCHSIZE)
    
    #Splits the data into training and testing
    train = dataset.take((1 - VALIDATIONSPLIT) * dataset.cardinality().numpy()) #Note: if we use filter, .cardinality will break
    test = dataset.skip((1 - VALIDATIONSPLIT) * dataset.cardinality().numpy())

    return train, test, ybase

#normalizes and loads data
#data: 1D array of filenames
#returns: array of images; images are of shape (x, y, colorchannels)
def normalize(data):
    images = []
    #for every img (file name) in data, do
    #load the actual image's pixel data
    #standardize its size, and normalize
    for img in data:
        # Read an image from a file
        image_string = tf.io.read_file('images/' + img)
        # Decode it into a dense vector
        image_decoded = tf.image.decode_jpeg(image_string, channels = 'rgb')
        # Resize it to fixed shape
        image_resized = tf.image.resize(image_decoded, IMAGESIZE)
        # Normalize it from [0, 255] to [0.0, 1.0]
        image_normalized = image_resized / 255.0

        images.append(image_normalized)

    return images
   
load_data()