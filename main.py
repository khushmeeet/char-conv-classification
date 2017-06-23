import tensorflow as tf
import helper
from sklearn.utils import shuffle

char_limit = 1014
n_classes = 3
num_feature_map = 256
nodes = 1024
epochs = 10

text, labels = helper.get_data()
vocab, vocab_size, char2idx, reverse_char2idx = helper.create_vocab_set(text)

text, labels = shuffle(text, labels)
encoded_text = helper.get_encoded_text(text, vocab_size, char2idx)
trainx , trainy, testx, testy = encoded_text[:3500], labels[:3500], encoded_text[:-3500], labels[:-3500]
train = helper.batch_gen(trainx, trainy, batch_size=128)
test = helper.batch_gen(testx, testy, batch_size=644)

x = tf.placeholder(dtype=tf.float32, shape=[None, char_limit, vocab_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
prob = tf.placeholder(dtype=tf.float32)

weights = {
    'conv7x7': tf.Variable(tf.truncated_normal([vocab_size, 7, 1, num_feature_map])),
    'conv3x3': tf.Variable(tf.truncated_normal([3, 3, num_feature_map])),
    'full1': tf.Variable(tf.truncated_normal([34*3, nodes])),
    'full2': tf.Variable(tf.truncated_normal([nodes, nodes])),
    'full3': tf.Variable(tf.truncated_normal([nodes, n_classes]))
}

biases = {
    'bias': tf.Variable(tf.truncated_normal([num_feature_map])),
    'full12': tf.Variable(tf.truncated_normal([nodes])),
    'full3': tf.Variable(tf.truncated_normal([n_classes]))
}

for ii,jj in train:
    print(ii[0].shape)
    print(jj)
    break

# layer 1
output = tf.nn.conv1d(x, weights['conv7x7'], stride=1, padding='VALID', name='conv1')
output = tf.nn.bias_add(output, biases['bias'])
output = tf.nn.relu(output)

output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=1, padding='VALID', name='pool1')

# layer 2
output = tf.nn.conv1d(output, weights['conv7x7'], stride=1, padding='VALID', name='conv2')
output = tf.nn.bias_add(output, biases['bias'])
output = tf.nn.relu(output)

output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=1, padding='VALID', name='pool2')

# layer 3
output = tf.nn.conv1d(output, weights['conv3x3'], stride=1, padding='VALID', name='conv3')
output = tf.nn.bias_add(output, biases['bias'])
output = tf.nn.relu(output)

# layer 4
output = tf.nn.conv1d(output, weights['conv3x3'], stride=1, padding='VALID', name='conv3')
output = tf.nn.bias_add(output, biases['bias'])
output = tf.nn.relu(output)

# layer 5
output = tf.nn.conv1d(output, weights['conv3x3'], stride=1, padding='VALID', name='conv3')
output = tf.nn.bias_add(output, biases['bias'])
output = tf.nn.relu(output)

# layer 6
output = tf.nn.conv1d(output, weights['conv7x7'], stride=1, padding='VALID', name='conv3')
output = tf.nn.bias_add(output, biases['bias'])
output = tf.nn.relu(output)

output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=1, padding='VALID', name='pool3')

# layer 7
output = tf.matmul(output, weights['full1']) + biases['full12']
output = tf.nn.dropout(output, prob)
# layer 8
output = tf.matmul(output, weights['full2']) + biases['full12']
output = tf.nn.dropout(output, prob)
# layer 9
output = tf.matmul(output, weights['full3']) + biases['full3']

logits = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
cost = tf.reduce_mean(logits)
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

step = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for xx, yy in train:
            print(xx)
            print(xx.shape)
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={x: xx, y:yy, prob:0.5})
            print('Batch {}  -  Loss {}  -  Acc {}'.format(step, loss, acc))
            step += 1
    for xx, yy in test:
        acc = sess.run(accuracy, feed_dict={x:xx, y:yy, prob:1.})
        print(acc)