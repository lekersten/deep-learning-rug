import tensorflow as tf
import cv2
import numpy as np


def pre_process():
    # Load CIFAR-10 dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # Convert images to LAB color space and normalize
    def convert_to_lab_and_normalize(images):
        L_channel = []
        AB_channels = []

        for img in images:
            # Convert to LAB color space
            lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

            # Normalize L channel to [0, 1]
            L = lab_img[:, :, 0] / 255.0
            L_channel.append(L)

            # Normalize AB channels to [-1, 1]
            AB = lab_img[:, :, 1:] / 128.0 - 1
            AB_channels.append(AB)

        return np.array(L_channel), np.array(AB_channels)

    # Process train and test images
    L_train, AB_train = convert_to_lab_and_normalize(x_train)
    L_test, AB_test = convert_to_lab_and_normalize(x_test)

    return L_train, AB_train, L_test, AB_test


def calculate_weights(AB_train, num_bins=313):
    # Combine all A and B channel values from the training dataset
    A_values = AB_train[:, :, :, 0].flatten()
    B_values = AB_train[:, :, :, 1].flatten()

    # Compute histograms for A and B channels
    A_hist, _ = np.histogram(A_values, bins=num_bins, range=(-1, 1))
    B_hist, _ = np.histogram(B_values, bins=num_bins, range=(-1, 1))

    # Normalize histograms to get the probability distribution
    A_prob = A_hist / np.sum(A_hist)
    B_prob = B_hist / np.sum(B_hist)

    # Calculate the inverse probability distribution as weights
    A_weights = 1 / (A_prob + 1e-8)
    B_weights = 1 / (B_prob + 1e-8)

    # Normalize the weights so that they sum to 1
    A_weights /= np.sum(A_weights)
    B_weights /= np.sum(B_weights)

    return A_weights, B_weights


def create_dataset(L, AB, num_bins=no_colour_bins, batch_size=32, buffer_size=10000, seed=None):
    # Create a dataset for the L channel
    L_dataset = tf.data.Dataset.from_tensor_slices(L)

    # Create a dataset for the AB channels
    AB_dataset = tf.data.Dataset.from_tensor_slices(AB)

    # Combine the L and AB datasets
    dataset = tf.data.Dataset.zip((L_dataset, AB_dataset))

    # Shuffle, batch, and prefetch the data
    dataset = dataset.shuffle(buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def load_data(no_colour_bins):
    L_train, AB_train, L_val, AB_val, L_test, AB_test = pre_process()

    # Calculate weights for A and B channels
    A_weights, B_weights = calculate_weights(AB_train, no_colour_bins)

    # Create the tf.data.Dataset objects for train, validation, and test sets
    train_dataset = create_dataset(L_train, AB_train, batch_size=32, buffer_size=10000, seed=42)
    test_dataset = create_dataset(L_test, AB_test, batch_size=32, buffer_size=10000, seed=42)

    return train_dataset, test_dataset, A_weights, B_weights
