###############################################################################
# Author: Kushal Kumar Kasina
# Description:
#	This code performs Principal Component Analysis (PCA) on MNIST Digits Data-
#	set. It only extracts ONE component out of the 28 possible components (whi-
#	ch happens to be the most dominant one).
#	The PCA is applied only on 59,000 images out of the 60,000 images availabl-
#	e as training set. The other 10,000 images are reserved for validating res-
#	ults on unseen data. PCA is (obviously) applied on the test set also.
###############################################################################

# required modules
import numpy as np						# array ops
from keras.datasets import mnist		# MNIST dataset
from sklearn.decomposition import PCA	# PCA algorithm
import time								# imma show you fast

# obtain mnist dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# keep aside last 1000 images from training data for unseen data accuracy valid-
# ation.
np.savetxt('X_unseenData.txt', X_train[59000:60000,:,:].reshape(1000, 784), delimiter=',')
np.savetxt('Y_unseenData.txt', Y_train[59000:60000], delimiter=',')

# array to store PCA results
X_pca = np.array([])

print("Processing Training Set...")

training_set_time = time.time()

# apply PCA to training set (59,000)
for m in range(X_train.shape[0]-1000):

	# PCA with options: 'full' SVD solver, and no whitening (removes dependancy 
	# on the 27 other eigen values)
	pca = PCA(n_components=1, svd_solver='full', whiten=False).\
		  fit(X_train[m,:,:])
	temp = pca.transform(X_train[m,:,:])

	# print progress
	print("Training Set: " + str(m))

	# store the feature in the array
	X_pca = np.append(X_pca, temp.astype('float32'))

training_set_time = time.time() - training_set_time

# reshape the PCA component array (array of features) for compatibility with ke-
# ras model input
X_train = X_pca.reshape(X_train.shape[0]-1000, 28)

# store all features in 'X_train_features.txt' with comma separation, & Y_train
np.savetxt('X_train_features.txt', X_train, delimiter=',')
np.savetxt('Y_train.txt', Y_train[0:59000], delimiter=',')

print("Finished Processing Training Set...")

# array to store PCA results
X_pca = np.array([])

print("Processing Test Set...")

test_set_time = time.time()

# apply PCA to test set (50,000)
for m in range(X_test.shape[0]):

	# PCA with options: 'full' SVD solver, and no whitening (removes dependancy 
	# on the 27 other eigen values
	pca = PCA(n_components=1, svd_solver='full', whiten=False).\
		  fit(X_test[m,:,:])
	temp = pca.transform(X_test[m,:,:])

	# print progress
	print("Test Set: " + str(m))

	# store the feature in the array
	X_pca = np.append(X_pca, temp.astype('float32'))

test_set_time = time.time() - test_set_time

# reshape the PCA component array (array of features) for compatibility with ke-
# ras model input
X_test = X_pca.reshape(X_test.shape[0], 28)

# store all features in 'X_test_features.txt' with comma separation, & Y_test
np.savetxt('X_test_features.txt', X_test, delimiter=',')
np.savetxt('Y_test.txt', Y_test, delimiter=',')

print("Finsihed Processing Test Set...")

print("Training Data Processing took " + str(training_set_time) + " seconds")
print("Test Data Processing took " + str(test_set_time) + " seconds")