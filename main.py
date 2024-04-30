# TensorFlow and tf.keras
import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# # print(train_images.shape)
# # plt.figure()
# # plt.imshow(train_images[0])
# # plt.colorbar()
# # plt.grid(False)
# # plt.show()
#
# # plt.figure(figsize=(8, 8))
# # for i in range(25):
# #     plt.subplot(5, 5, i + 1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(train_images[i], cmap=plt.cm.binary)
# #     plt.xlabel(class_names[train_labels[i]])
# # plt.show()
#

# Model 1
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(36, activation='relu'),
#     # keras.layers.Dropout(0.2),
#     keras.layers.Dense(10)
# ])
#
# model.compile(optimizer='adam', metrics=['accuracy'],
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
#
# model.fit(train_images, train_labels, epochs=10, validation_split=0.1, shuffle=True, verbose=2)
#
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'loss : {loss}, accuracy : {acc}')
#
# probability_model = tf.keras.Sequential([
#     model,
#     keras.layers.Softmax()
# ])
#
# probability_model.build()
# probability_model.save('model1.keras')

probability_model = keras.models.load_model('model1.keras')
print(probability_model.summary())
pred = probability_model.predict(test_images)


#
# print('First pred : ', pred[0], ' ', np.argmax(pred[0]))


def plot_image(i, prediction_array, true_labels, images):
    true_label, img = true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted = np.argmax(prediction_array)
    if predicted == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{class_names[predicted]} {100 * np.max(prediction_array):2.0f}% ({class_names[true_label]})",
               color=color)


def plot_value_array(i, predictions_array, true_labels):
    true_label = true_labels[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, pred[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, pred[i], test_labels)
plt.tight_layout()
plt.show()
# TensorFlow and tf.keras
import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# # print(train_images.shape)
# # plt.figure()
# # plt.imshow(train_images[0])
# # plt.colorbar()
# # plt.grid(False)
# # plt.show()
#
# # plt.figure(figsize=(8, 8))
# # for i in range(25):
# #     plt.subplot(5, 5, i + 1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(train_images[i], cmap=plt.cm.binary)
# #     plt.xlabel(class_names[train_labels[i]])
# # plt.show()
#

# Model 1
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(36, activation='relu'),
#     # keras.layers.Dropout(0.2),
#     keras.layers.Dense(10)
# ])
#
# model.compile(optimizer='adam', metrics=['accuracy'],
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
#
# model.fit(train_images, train_labels, epochs=10, validation_split=0.1, shuffle=True, verbose=2)
#
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'loss : {loss}, accuracy : {acc}')
#
# probability_model = tf.keras.Sequential([
#     model,
#     keras.layers.Softmax()
# ])
#
# probability_model.build()
# probability_model.save('model1.keras')

probability_model = keras.models.load_model('model1.keras')
print(probability_model.summary())
pred = probability_model.predict(test_images)


#
# print('First pred : ', pred[0], ' ', np.argmax(pred[0]))


def plot_image(i, prediction_array, true_labels, images):
    true_label, img = true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted = np.argmax(prediction_array)
    if predicted == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{class_names[predicted]} {100 * np.max(prediction_array):2.0f}% ({class_names[true_label]})",
               color=color)


def plot_value_array(i, predictions_array, true_labels):
    true_label = true_labels[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, pred[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, pred[i], test_labels)
plt.tight_layout()
plt.show()
