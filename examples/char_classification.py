import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification

from keras.preprocessing.image import ImageDataGenerator
train_dir = 'E:\\Datasets\\bangla\\train'
test_dir = 'E:\\Datasets\\bangla\\test'


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (64, 64),
    batch_size=12000
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (64, 64),
    batch_size=3000
)

for X_train, Y_train in train_generator:
    X_train = X_train.reshape(-1, 64*64*3)
    break

for X_test, Y_test in validation_generator:
    X_train = X_train.reshape(-1, 64*64*3)
    break

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)
classifier.save('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))


