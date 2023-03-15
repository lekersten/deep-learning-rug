import tensorflow as tf

def ssim_score(y_pred, y_true):
    return tf.reduce_mean(tf.image.ssim(y_pred, y_true, max_val=1.0))

def ssim_loss(y_pred, y_true):
    return 1 - ssim_score(y_pred, y_true)
	
	
def mae_loss(y_pred, y_true):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))