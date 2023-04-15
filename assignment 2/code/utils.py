import tensorflow as tf
from tensorflow.image import ssim
from tensorflow.math import reduce_mean
import matplotlib.pyplot as plt

class WeightedSSIMLoss(tf.keras.losses.Loss):
    def __init__(self, A_weights, B_weights, max_val=1.0, **kwargs):
        super().__init__(**kwargs)
        self.A_weights = tf.cast(A_weights, tf.float32) # Convert A_weights to float32
        self.B_weights = tf.cast(B_weights, tf.float32) # Convert B_weights to float32
        self.max_val = max_val

    def call(self, y_true, y_pred):
        # Separate the A and B channels from true and predicted values
        y_true_A = y_true[..., 0]
        y_true_B = y_true[..., 1]
        y_pred_A = y_pred[..., 0]
        y_pred_B = y_pred[..., 1]

        # Calculate SSIM for A and B channels separately
        A_ssim = ssim(y_true_A[..., None], y_pred_A[..., None], max_val=self.max_val)
        B_ssim = ssim(y_true_B[..., None], y_pred_B[..., None], max_val=self.max_val)

        # Bin the A and B channel values
        A_bins = tf.cast((y_true_A + 1) * (no_colour_bins - 1) / 2, tf.int32)
        B_bins = tf.cast((y_true_B + 1) * (no_colour_bins - 1) / 2, tf.int32)

        # Gather corresponding weights from A_weights and B_weights
        A_sample_weights = tf.gather(self.A_weights, A_bins)
        B_sample_weights = tf.gather(self.B_weights, B_bins)

        # Calculate the weighted SSIM loss
        weighted_A_loss = reduce_mean((1.0 - A_ssim) * A_sample_weights)
        weighted_B_loss = reduce_mean((1.0 - B_ssim) * B_sample_weights)

        # Combine the weighted losses for both channels
        total_weighted_loss = weighted_A_loss + weighted_B_loss

        return total_weighted_loss

class WeightedMeanAbsoluteError(tf.keras.losses.Loss):
    def __init__(self, A_weights, B_weights, **kwargs):
        super().__init__(**kwargs)
        self.A_weights = tf.cast(A_weights, tf.float32)
        self.B_weights = tf.cast(B_weights, tf.float32)

    def call(self, y_true, y_pred):
        # Separate the A and B channels from true and predicted values
        y_true_A = y_true[..., 0]
        y_true_B = y_true[..., 1]
        y_pred_A = y_pred[..., 0]
        y_pred_B = y_pred[..., 1]

        # Calculate the absolute error for A and B channels
        A_error = tf.abs(y_true_A - y_pred_A)
        B_error = tf.abs(y_true_B - y_pred_B)

        # Bin the A and B channel values
        A_bins = tf.cast((y_true_A + 1) * (no_colour_bins - 1) / 2, tf.int32)
        B_bins = tf.cast((y_true_B + 1) * (no_colour_bins - 1) / 2, tf.int32)

        # Gather corresponding weights from A_weights and B_weights
        A_sample_weights = tf.gather(self.A_weights, A_bins)
        B_sample_weights = tf.gather(self.B_weights, B_bins)

        # Calculate the weighted mean absolute error
        weighted_A_error = tf.reduce_mean(A_error * A_sample_weights)
        weighted_B_error = tf.reduce_mean(B_error * B_sample_weights)

        # Combine the weighted errors for both channels
        total_weighted_error = weighted_A_error + weighted_B_error

        return total_weighted_error


def display(model, dataset, n_images):
    for L, AB in dataset.take(n_images):
        L = L.numpy()
        AB = AB.numpy()

        # Unnormalize L channel and AB channels
        L_unnorm = (L * 255.0).astype(np.uint8)
        AB_unnorm = ((AB + 1) * 128.0).astype(np.uint8)

        # Run model prediction
        AB_pred = model.predict(L)
        AB_pred_unnorm = ((AB_pred + 1) * 128.0).astype(np.uint8)

        # Display images
        for i in range(n_images):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

            # Display L channel image
            ax1.imshow(L_unnorm[i], cmap='gray')
            ax1.set_title("L Channel")
            ax1.axis('off')

            # Display ground truth image
            lab_img_gt = np.concatenate((L_unnorm[i][:, :, np.newaxis], AB_unnorm[i]), axis=2)
            rgb_img_gt = cv2.cvtColor(lab_img_gt, cv2.COLOR_LAB2RGB)
            ax2.imshow(rgb_img_gt)
            ax2.set_title("Ground Truth")
            ax2.axis('off')

            # Display predicted image
            lab_img_pred = np.concatenate((L_unnorm[i][:, :, np.newaxis], AB_pred_unnorm[i]), axis=2)
            rgb_img_pred = cv2.cvtColor(lab_img_pred, cv2.COLOR_LAB2RGB)
            ax3.imshow(rgb_img_pred)
            ax3.set_title("Predicted")
            ax3.axis('off')

            plt.show()

def kfold_cv(model, train_ds, loss, epochs, model_str, loss_str, k=5):

    num_samples = dataset.reduce(0, lambda x, _: x + 1).numpy()
    fold_size = num_samples // k

    # Loop through each fold
    for fold in range(k):
        print(f'Fold: {fold + 1}')

        # Calculate the start and end indices for the training and validation sets
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size
        val_idx = tf.range(start_idx, end_idx)
        train_idx = tf.concat([tf.range(0, start_idx), tf.range(end_idx, num_samples)], axis=0)

        # Split the data into training and validation sets
        train_dataset = dataset.enumerate().filter(lambda i, x_y: i not in val_idx).map(lambda i, x_y: x_y).batch(
            batch_size)
        val_dataset = dataset.enumerate().filter(lambda i, x_y: i in val_idx).map(lambda i, x_y: x_y).batch(batch_size)

        # Define your optimizer, loss function, and evaluation metric
        optimizer = tf.keras.optimizers.Adam()

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss)

        # Train the model on the training set
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

        model.save_weights(f"models/{model_str}_{loss_str}_{fold+1}.h5")

        # Save the history to a dictionary
        fold_history = {
            'fold': fold + 1,
            'train_loss': history.history['loss'],
            'train_accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy']
        }

        # Append the history to the list
        history_list.append(fold_history)

    # Combine the history for each fold into a single DataFrame
    history_df = pd.concat([pd.DataFrame(h) for h in history_list])

    # Save the DataFrame to a CSV file
    history_df.to_csv(f'loss/history_{model_str}_{loss_str}.csv', index=False)


