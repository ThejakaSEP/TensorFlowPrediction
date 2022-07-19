import os
import tensorflow as tf
# tf.disable_v2_behavior()

# Turn off TEnsorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()
# Define computational graph
X = tf.compat.v1.placeholder(tf.float32,name = "X")
Y = tf.compat.v1.placeholder(tf.float32,name = "Y")

addition = tf.add(X,Y,name="Addition")

# Create the session

with tf.compat.v1.Session() as session:
    result = session.run(addition,feed_dict={X:[1,2,14],Y:[4,2,3]})

    print(result)
