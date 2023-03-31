import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#This is the data modelling for clothes dataset- fashion_mnist (a dataset containing 60000 cloth images for training)

clothing_fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = clothing_fashion_mnist.load_data()


print('Shape of training cloth images: ', x_train.shape)

print('Shape of training label: ',
	y_train.shape)

print('Shape of test cloth images: ',
	x_test.shape)

print('Shape of test labels: ',
	y_test.shape)

label_class_names = ['T-shirt/top', 'Trouser',
                     'Pullover', 'Dress', 'Coat',
                     'Sandal', 'Shirt', 'Sneaker',
                     'Bag', 'Ankle boot']

plt.imshow(x_train[0]) 
plt.colorbar() 

x_train = x_train / 255.0 
x_test = x_test / 255.0 

plt.figure(figsize=(15, 5)) 
i = 0
while i < 20:
	plt.subplot(2, 10, i+1)
	
	
	plt.imshow(x_train[i], cmap=plt.cm.binary)
	
	
	plt.xlabel(label_class_names[y_train[i]])
	i = i+1



model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10)
])


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test,y_test,verbose=2)
print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)



prediction_model = tf.keras.Sequential(
	[model, tf.keras.layers.Softmax()])



prediction = prediction_model.predict(x_test)


print('Predicted test label:', np.argmax(prediction[0]))


print(label_class_names[np.argmax(prediction[0])])


print('Actual test label:', y_test[0])


plt.figure(figsize=(15, 6))
i = 0

while i<24:
	image, actual_label = x_test[i], y_test[i]
	predicted_label= np.argmax(prediction[i])
	plt.subplot(3,8,i+1)
	plt.tight_layout()
	plt.xticks([])
	plt.yticks([])
	plt.imshow(image)
	if predicted_label==actual_label:
		color, label= ("green", "Correct Prediction")
	else:
		color, label= ("red", "Incorrect Prediction")
	plt.title(label, color=color)
	plt.xlabel(" {} ~ {} ".format(label_class_names[actual_label], label_class_names[predicted_label]))
	plt.ylabel(i)
	i+=1

plt.show()
model.save("clothing_classifier.h5")
