import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
warnings.filterwarnings('ignore')

# split dataset
def process_fnn_data(X, Y):
    '''
    X: Image data
    Y: Label data
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state = 0)
    #print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape) # (1752, 64, 64) (310, 64, 64) (1752, 10) (310, 10)
    num_train = X_train.shape[0] # 1752
    num_test = X_test.shape[0] # 310

    # flat image to prepare for FNN model input
    X_train_flat = X_train.reshape(num_train, pixels)
    X_test_flat = X_test.reshape(num_test, pixels)
    #print("After flattening X_train: ", X_train_flat.shape,"\n","After flattening X_test: ", X_test_flat.shape)
    return X_train_flat, Y_train, X_test_flat, Y_test

# Define vanilla feedforward model
def build_FNN_model(learn_rate = 0.001, momentum=0, activation='relu', optimizer='SGD'):
    fnn_model = Sequential()
    fnn_model.add(Dense(128, kernel_initializer='uniform', activation=activation, input_shape=(pixels,)))
    fnn_model.add(Dense(128, kernel_initializer='uniform', activation=activation))
    fnn_model.add(Dense(128, kernel_initializer='uniform', activation=activation))
    fnn_model.add(Dense(10, activation='softmax'))
    #opt = SGD(lr=learn_rate,momentum=momentum)
    fnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return fnn_model

def tune_parameter_gridsearch(model, X_train, X_test, Y_train, Y_test):
    '''
    model: model to fine tune (model building function)
    X_train: training data from data processing function
    X_test: validation data
    Y_train: training label
    Y_test: validation label
    '''
    # define search grid
    #learn_rate_opts = [0.0001, 0.001, 0.01]
    #momentum_opts = [0.0, 0.4, 0.8, 0.9, 0.95, 0.99]
    activation_opts = ['relu', 'softmax', 'sigmoid', 'linear', 'tanh']
    optimizer_opts = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam']
    batch_size_opts = [32, 64, 128, 256]
    epochs_opts = [50, 100, 150, 200]
    param_grid = {
        #'learn_rate': learn_rate_opts,
        #'momentum': momentum_opts,
        'activation': activation_opts,
        'optimizer': optimizer_opts,
        'epochs': epochs_opts,
        'batch_size': batch_size_opts
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)
    grid_result = grid_search.fit(X_train, Y_train, validation_data=(X_test, Y_test))

    # result
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def process_cnn_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state = 0)
    # reshape image to (64, 64, 1) to prepare for 
    X_train = X_train.reshape((X_train.shape[0], 64, 64, 1))
    X_test = X_test.reshape((X_test.shape[0], 64, 64, 1))

    return X_train, Y_train, X_test, Y_test

def build_cnn_model(activation='relu', dropout_rate = 0, optimizer='Adam'):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (3,3), activation=activation, input_shape=(img_size, img_size, 1)))
    #cnn_model.add(Conv2D(32, (3,3), activation='relu'))
    cnn_model.add(MaxPool2D((2,2)))
    cnn_model.add(Dropout(dropout_rate))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation=activation))
    cnn_model.add(Dropout(dropout_rate))
    cnn_model.add(Dense(10, activation='softmax'))
    cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn_model

# load data
x_1 = np.load('/Users/justinlim/Desktop/sign_lang_digit_recognition/input/Sign-language-digits-dataset/X.npy')
Y_1 = np.load('/Users/justinlim/Desktop/sign_lang_digit_recognition/input/Sign-language-digits-dataset/Y.npy')
print('x shape:',x_1.shape) # (2062, 64, 64)
print('y shape:', Y_1.shape) # (2062, 10)

# create some global variables
img_size = 64
pixels = img_size * img_size # 64 * 64 = 4096

## Look at images
# create new label by decoding one hot encoded labels
label_new = []
for target in Y_1:
    label_new.append(np.argmax(target))
label = np.array(label_new)

# images and labels are mismatched
# correctly assign images to the corresponding labels
label_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}
label_new = []
for s in label:
    label_new.append(label_map[s])
label = np.array(label_new)

n_class = 10
n_samples = 5
fig, ax = plt.subplots(nrows=n_class, ncols=n_samples, figsize=(18, 18))
ax = ax.flatten()
plt_id = 0
for sign in range(n_class):
    # create indexes for each class
    sign_indexes = np.where(label==sign)[0]
    # show 5 images for each class
    for i in range(n_samples):
        image_index = sign_indexes[i]
        ax[plt_id].imshow(x_1[image_index])
        ax[plt_id].set_xticks([])
        ax[plt_id].set_yticks([])
        ax[plt_id].set_title("Sign :{}".format(sign))
        plt_id += 1

plt.suptitle('{} Sample for each classes'.format(n_samples))
plt.show()

# process data and define model for FNN 
X_train_flat, Y_train_fnn, X_test_flat, Y_test_fnn = process_fnn_data(x_1, Y_1)
fnn_model = KerasClassifier(build_fn=build_FNN_model)
# Comment out when grid search is not neccessary
tune_parameter_gridsearch(fnn_model, X_train_flat, X_test_flat, Y_train, Y_test)
#Best: 0.783105 using {'activation': 'relu', 'batch_size': 32, 'epochs': 200, 'optimizer': 'Adamax'}

# process data and define model for CNN
X_train_cnn, Y_train_cnn, X_test_cnn, Y_test_cnn = process_cnn_data(x_1, Y_1)
cnn_model = KerasClassifier(build_fn=build_cnn_model)
tune_parameter_gridsearch(cnn_model, X_train_cnn, X_test_cnn, Y_train_cnn, Y_test_cnn)
