import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.nn.dynamic_rnn as dynamic_rnn

class CharRNNPrediction:

	def __init__(self, num_classes, batch_size=64, num_steps= 50, sampling = False, num_layers = 2, lstm_size=128):
		# Testing/Training
		batch_size, num_steps = state(sampling, batch_size, num_steps)

		# Define inputs/outputs on graph
		self.inputs, self.targets, self.keep_prob = set_placeholders(batch_size, num_steps)

		#Define LSTM cells in model
		cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

		#one hot encoder
		x_one_hot = tf.one_hot(self.inputs, num_classes)

		#run
		outputs, state = dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
		self.initial_state = state

		#output
		self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)



	def build_output(outputs, lstm_size, num_classes):



	def build_lstm(lstm_size, num_layers, batch_size, keep_prob):

		def build_cell(lstm_size, keep_prob):
			lstm = rnn.BasicLSTMCell(lstm_size)

			drop = rnn.DropoutWrapper(lstm, output_keep_prob= keep_prob)

			return drop

		cell = rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
		initial_state = cell.zero_state(batch_size, tf.float32)

		return cell, initial_state

	def set_placeholders(batch_size, num_steps):
		#Input/train
		input = tf.placeholder(tf.int32, [batch_size, num_steps])
		#output/train
		output = tf.placeholder(tf.int32, [batch_size, num_steps])

		keep_prob = tf.placeholder(tf.float32)

		return input, output, keep_prob

	def state(sampling, batch_size, num_steps):
		if sampling:
			return 1, 1
		else:
			return batch_size, num_steps
