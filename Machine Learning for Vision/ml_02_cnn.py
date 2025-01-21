#TensorFlow is an open-source machine learning library developed by Google. 
# TensorFlow is used to build and train deep learning models as 
# it facilitates the creation of computational graphs and 
# efficient execution on various hardware platforms. 
# The article provides an comprehensive overview of tensorflow.

import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Load and preprocess the CIFAR-10 dataset
(train_image, train_labels), (test_image, test_labels) = datasets.cifar10.load_data()
train_image, test_image = train_image / 255.0, test_image / 255.0

# Create the CNN model-> 
# A convolutional neural network (CNN) is a machine learning model 
# that analyzes and recognizes patterns in images and videos. 
# CNNs are a type of artificial neural network that are used in computer vision tasks.  

models = models.Sequential()
models.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
models.add(layers.MaxPooling2D((2, 2)))
models.add(layers.Conv2D(64, (3, 3), activation='relu'))
models.add(layers.MaxPooling2D((2, 2)))
models.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add Dense layers on top
models.add(layers.Flatten())
models.add(layers.Dense(64, activation='relu'))
models.add(layers.Dense(10))
# models.summary()

# Train the model
models.fit(train_image, train_labels, epochs=10, validation_data=(test_image, test_labels))

# Evaluate the model
test_loss, test_acc = models.evaluate(test_image, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')



"""How CNNs work 
CNNs are made up of layers that process input data.
Convolutional layers: Apply filters to extract features from the input.
Pooling layers: Reduce the spatial dimensions of the features.
Fully connected layers: Connect the extracted features to the final output.
Training: CNNs are trained with labeled data, adjusting weights to optimize performance.
Applications of CNNs 
Image classification: CNNs are used to classify images.
Object detection: CNNs are used to detect objects in images and videos.
Image segmentation: CNNs are used to segment images.
Natural language processing: CNNs are used to process natural language.
Drug discovery: CNNs are used in drug discovery.
Health risk assessments: CNNs are used to assess health risks.
Depth estimation: CNNs are used to estimate depth for self-driving cars.
Challenges with CNNs
CNNs require millions of labeled data points for training. 
CNNs can be computationally demanding, requiring graphical processing units (GPUs) to train models. """
