import input_data
mnist=input_data.read_data_sets("../../data/",one_hot=True)
import tensorflow as tf
n_hidden_1 = 256
n_hidden_2 = 256

learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 1

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W" ,weight_shape,initializer = weight_init)
    b = tf.get_variable("b",bias_shape,initializer = bias_init)
    return tf.nn.relu(tf.matmul(input,W)+b)
def interference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [784, n_hidden_1],[n_hidden_1])
        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1,[n_hidden_1,n_hidden_2],[n_hidden_2])
            with tf.variabl_scope("ouyput"):
                output = layer(hidden_2,[n_hidden_2,10],[10])
                return output
def loss(output,y):
    xentropy = tf.nn.softmax_cross_entropy_with_logistic(logits =output, lables=y)
    loss =tf.reduce_mean(xentropy)
    return loss
def training(cost,global_step):
    tf.summary.scalar("cost",cost)
    optimizer = tf.train.GradientDescentoptimizer(learning_rate)
    train_op =optimizer.minimize(cost,global_step =global_step)
    return train_op
def evaluate(output,y):
    correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("validation",accuracy) 
    return accuracy    