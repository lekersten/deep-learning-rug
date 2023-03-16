import tensorflow as tf


def data_loader():
    # Load Cifar10 dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # label data is coloured images, train data is black and white images
    y_train = x_train
    y_test = x_test

    # Convert x_train and x_test to black and white
    x_train = tf.image.rgb_to_grayscale(x_train)
    x_test = tf.image.rgb_to_grayscale(x_test)

    # Load to Tensorflow Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Shuffle and batch with tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(10000).batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.shuffle(10000).batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds