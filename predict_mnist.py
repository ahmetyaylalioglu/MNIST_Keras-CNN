from os import environ
from keras.models import load_model
from keras.datasets import mnist
from numpy import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical

environ['TF_CPP_MIN_LOG_LEVEL']='3'

#Loading Model
my_model = load_model(filepath=r'C:\Users\user\Desktop\CNNudemy\handnumbers\model_save.h5')
print(my_model.summary(),'\n')

print("Last node biases: \n")
print(my_model.get_weights()[-1])
print("Last node weights: \n")
print(my_model.get_weights()[-2])

#Loading dataset
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
rand_n = random.randint(0,100)
rand_i = test_images[rand_n]
plt.imshow(rand_i,cmap='Greys')
plt.show()

#Predicting a random number Image
pred = my_model.predict(rand_i.reshape(1,28,28,1),batch_size=1)
print('The Random number Image Prediction is : {}'.format(pred))

test_images = test_images.reshape((10000,28,28,1))
test_labels = to_categorical(test_labels)

#Evaluation over the entire dataset
(eval_loss,eval_accuracy)=my_model.evaluate(x = test_images,y=test_labels,batch_size=10000)
print("Evaluation Accuracy is: {:4.2f} ".format(eval_accuracy*100))
