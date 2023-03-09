import tensorflow as tf

def r2(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + tf.keras.backend.epsilon()) )