import tensorflow as tf


def loss_custom(y_true, y_pred):
    metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True )
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = metric(y_true, y_pred, sample_weight =mask)
    return loss

def accuracy(predicted, actual):
    counter_true = 0
    counter_all = 0
    for i in range(len(predicted)):
        counter_all +=1
        if actual[i]==predicted[i]:
            counter_true +=1
    accuracy_result = counter_true / float(len(actual)) *100
    return accuracy_result