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

# load data
x_1 = np.load('/Users/justinlim/Desktop/sign_lang_digit_recognition/input/Sign-language-digits-dataset/X.npy')
Y_1 = np.load('/Users/justinlim/Desktop/sign_lang_digit_recognition/input/Sign-language-digits-dataset/Y.npy')
print('x shape:',x_1.shape) # (2062, 64, 64)
print('y shape:', Y_1.shape) # (2062, 10)

# looking at data
# plt.subplot(1,2,1)
# # digit 0
# plt.imshow(x_1[260].reshape(img_size, img_size))
# plt.axis('off')

# # digit 1
# plt.subplot(1,2,2)
# plt.imshow(x_1[900].reshape(img_size, img_size))
# plt.axis('off')
# plt.show()

# Flatten images
img_size = 64
pixels = img_size * img_size # 64 * 64 = 4096

# split dataset
def process_fnn_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state = 0)
    #print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape) # (1752, 64, 64) (310, 64, 64) (1752, 10) (310, 10)
    num_train = X_train.shape[0] # 1752
    num_test = X_test.shape[0] # 310

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
        #'epochs': epochs_opts,
        #'batch_size': batch_size_opts
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

# Comment out when grid search is not neccessary
tune_parameter_gridsearch(model, X_train_flat, X_test_flat, Y_train, Y_test)
#Best: 0.783105 using {'activation': 'relu', 'batch_size': 32, 'epochs': 200, 'optimizer': 'Adamax'}


X_train_flat, Y_train, X_test_flat, Y_test = process_fnn_data(x_1, Y_1)
model = build_FNN_model(optimizer='Adamax')
model.fit(X_train_flat, Y_train, batch_size=32, epochs=200, verbose=1, validation_data=(X_test_flat, Y_test))
loss, acc = model.evaluate(X_test_flat, Y_test)
print(loss, acc) # 0.9732992053031921, 0.8032258152961731

# X_train_flat, Y_train, X_test_flat, Y_test = process_fnn_data(x_1, Y_1)

# def build_keras_base(hidden_layers=[128,128,128], dropout_rate=0, l2_penalty=0.1, activation='relu', lr=0.001, momentum = 0.95, n_input=pixels, n_class=10):
#     model=Sequential()
#     for index, layers in enumerate(hidden_layers):
#         if not index:
#             model.add(Dense(layers, input_dim=pixels, kernel_regularizer=l2(l2_penalty), activation=activation))
#         else:
#             model.add(Dense(layers, kernel_regularizer=l2(l2_penalty), activation=activation))
#         if dropout_rate:
#             model.add(Dropout(dropout_rate))

#     model.add(Dense(n_class, activation='softmax'))
#     loss='categorical_crossentropy'
#     model.compile(loss=loss, optimizer=SGD(learning_rate=lr, momentum=momentum), metrics=['accuracy'])

#     return model

# callbacks=[EarlyStopping(
#     monitor='val_loss', min_delta=0.01, patience=5, verbose=0
# )]



# model_keras = KerasClassifier(
#     build_fn=build_keras_base,
#     n_input = pixels,
#     n_class = 10
# )


# fit_params = {
#     #'callbacks': callbacks,
#     'epochs': 200,
#     'batch_size': 1024,
#     'validation_data': (X_test_flat, Y_test),
#     'verbose': 1
# }

# # random search parameters
# dropout_rates_opt = [0, 0.25, 0.5]
# #hidden_layers_opt = [[128, 128, 128], [64,64,64,64], [32,32,32,32,32]]
# # l2_penalty= [0.01, 0.1, 0.2, 0.5]
# lr_opt = [0.001, 0.01, 0.1]
# momentum_opt = [0.9, 0.95, 0.99]
# keras_param_opts = {
#     #'hidden_layers': hidden_layers_opt,
#     'dropout_rate': dropout_rates_opt,
#     #'l2_penalty': l2_penalty,
#     'lr': lr_opt,
#     'momentum': momentum_opt
# }

# random_search = RandomizedSearchCV(
#     model_keras,
#     param_distributions=keras_param_opts,
#     scoring='neg_log_loss',
#     cv=3,
#     n_jobs=-1,
#     verbose=1
# )

# random_search.fit(X_train_flat, Y_train, **fit_params)

# best_model = random_search.best_estimator_.model
# #metric_names = best_model.metric_names
# loss, acc = best_model.evaluate(X_test_flat, Y_test)
# # for metric, value in zip(metric_names, metric_values):
# #     print('\n', metric, ': ', value)
# print(loss, acc)


# print('Best Score Obatained: {0}'.format(random_search.best_score_))
# print('Parameters:')
# for param, value in random_search.best_params_.items():
#     print('\t{}: {}'.format(param,value))

# evaluate
def evaluate_fnn_model(X_train, Y_train, X_test, Y_test):
    fnn_model = build_FNN_model()
    fnn_model.fit(X_train, Y_train, batch_size=128, epochs=150, validation_data=(X_test, Y_test), verbose=1)
    loss, acc = fnn_model.evaluate(X_test, Y_test, verbose=1)

    return loss, (acc*100)

def run_fnn_model():
    X_train_flat, Y_train, X_test_flat, Y_test = process_fnn_data(x_1, Y_1)
    fnn_loss, fnn_acc = evaluate_fnn_model(X_train_flat, Y_train, X_test_flat, Y_test)
    return fnn_loss, fnn_acc

# fnn_loss, fnn_acc = run_fnn_model()
# print('fnn_loss: ',fnn_loss,'\n', 'fnn_acc: ', fnn_acc)

def process_cnn_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state = 0)
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

def evaluate_cnn_model(X_train, Y_train, X_test, Y_test):
    cnn_model = build_cnn_model()
    cnn_model.fit(X_train, Y_train, batch_size=32, epochs=30, validation_data=(X_test, Y_test), verbose=1)
    loss, acc = cnn_model.evaluate(X_test, Y_test, verbose=1)

    return loss, (acc*100)

def run_cnn_model():
    X_train, Y_train, X_test, Y_test = process_cnn_data(x_1, Y_1)
    cnn_loss, cnn_acc = evaluate_cnn_model(X_train, Y_train, X_test, Y_test)
    return cnn_loss, cnn_acc

# cnn_loss, cnn_acc = run_cnn_model()
# print('cnn_loss: ', cnn_loss,'\n', 'cnn_acc: ', cnn_acc)

# X_train, Y_train, X_test, Y_test = process_cnn_data(x_1, Y_1)

# cnn_model = KerasClassifier(build_fn=build_cnn_model, epochs=30, batch_size=64)

# activation_opts = ['relu', 'softmax', 'sigmoid', 'linear', 'tanh']
# dropout_rate_opts = [0.0, 0.25, 0.5]
# optimizer_opts = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam']
# param_grid = {
#     'activation': activation_opts,
#     'dropout_rate': dropout_rate_opts,
#     'optimizer': optimizer_opts
# }

# grid_search = GridSearchCV(estimator=cnn_model, param_grid=param_grid, n_jobs=-1, cv = 3, verbose=1)
# grid_result = grid_search.fit(X_train, Y_train, validation_data=(X_test, Y_test))

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))