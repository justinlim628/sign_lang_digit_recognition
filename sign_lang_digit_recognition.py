import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# load data
x_1 = np.load('/Users/justinlim/Desktop/sign_lang_digit_recognition/input/Sign-language-digits-dataset/X.npy')
Y_1 = np.load('/Users/justinlim/Desktop/sign_lang_digit_recognition/input/Sign-language-digits-dataset/Y.npy')

# looking at data
img_size = 64
plt.subplot(1,2,1)
# digit 0
plt.imshow(x_1[260].reshape(img_size, img_size))
plt.axis('off')

# digit 1
plt.subplot(1,2,2)
plt.imshow(x_1[900].reshape(img_size, img_size))
plt.axis('off')
plt.show()

print('x shape:',x_1.shape) # (2062, 64, 64)
print('y shape:', Y_1.shape) # (2062, 10)

print('Example Label looks like: ',Y_1[9]) # one hot encoded

# split dataset
X_train, X_test, Y_train, Y_test = train_test_split(x_1, Y_1, test_size=0.15, random_state = 0)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape) # (1752, 64, 64) (310, 64, 64) (1752, 10) (310, 10)
num_train = X_train.shape[0] # 1752
num_test = X_test.shape[0] # 310

# Flatten iamges
pixels = img_size * img_size # 64 * 64 = 4096
X_train_flat = X_train.reshape(num_train, pixels)
X_test_flat = X_test.reshape(num_test, pixels)
print("After flattening X_train: ", X_train_flat.shape,"\n","After flattening X_test: ", X_test_flat.shape)


