import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.util import random_noise
from keras.datasets import fashion_mnist 
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D



(train_images, _), (test_images, _) = fashion_mnist.load_data()  


scaler = MinMaxScaler()
train_images_normalized = scaler.fit_transform(train_images.reshape(-1, 784))
test_images_normalized = scaler.transform(test_images.reshape(-1, 784))


train_images_normalized = train_images_normalized.reshape(-1, 28, 28, 1)
test_images_normalized = test_images_normalized.reshape(-1, 28, 28, 1)


noise_factor = 0.2
train_noisy_images = random_noise(train_images_normalized, var=noise_factor**2)
test_noisy_images = random_noise(test_images_normalized, var=noise_factor**2)


train_noisy_images = np.clip(train_noisy_images, 0., 1.)
test_noisy_images = np.clip(test_noisy_images, 0., 1.)

def plot_images(original_images, noisy_images, num_images=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Plot original images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Plot noisy images
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(noisy_images[i].reshape(28, 28), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


plot_images(train_images, train_noisy_images)


input_shape = (28, 28, 1)


input_img = Input(shape=input_shape)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

autoencoder.fit(train_noisy_images, train_images_normalized,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(test_noisy_images, test_images_normalized))

def plot_clean_images(clean_images, num_images=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(clean_images[i].reshape(28, 28), cmap='gray')
        plt.title("Clean")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


plot_clean_images(test_images_normalized)

from skimage.metrics import mean_squared_error, structural_similarity
import numpy as np


predicted_images = autoencoder.predict(test_noisy_images)


mse = mean_squared_error(test_images_normalized, predicted_images)
print("Mean Squared Error (MSE):", mse)


ssim = structural_similarity(test_images_normalized, predicted_images, multichannel=False)
print("Structural Similarity Index (SSIM):", ssim)


def plot_reconstructed_images(original_images, reconstructed_images, num_images=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Plot original images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Plot reconstructed images
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_reconstructed_images(test_images_normalized, predicted_images)

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

def create_autoencoder(learning_rate=0.001, batch_size=128, sparsity_reg=0.001):
    # Define input shape
    input_shape = (28, 28, 1)

    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    
    return autoencoder

# Create a KerasRegressor based on the autoencoder model function
autoencoder_regressor = KerasRegressor(build_fn=create_autoencoder, epochs=10, verbose=0)

# Define hyperparameters grid for grid search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'sparsity_reg': [0.001, 0.01, 0.1]
}

# Perform grid search
grid_search = GridSearchCV(estimator=autoencoder_regressor, param_grid=param_grid, cv=3)
grid_search_result = grid_search.fit(train_noisy_images, train_images_normalized)

# Print best hyperparameters
print("Best Parameters: ", grid_search_result.best_params_)


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout
from keras import regularizers

# Define input shape
input_shape = (28, 28, 1)

# Encoder
input_img = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Print model summary
autoencoder.summary()