import tensorflow as tf
import numpy as np

input = tf.placeholder(tf.int32, [4])

input_endcoded = tf.one_hot(input, 5)

init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)
	out = session.run(input_endcoded, feed_dict={input: [10,2,4,3]})
	print(out)