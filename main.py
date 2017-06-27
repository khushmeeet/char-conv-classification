import tensorflow as tf
import helper
from sklearn.utils import shuffle


sent_limit = 200
n_classes = 3
num_feature_map = 256
nodes = 1024
epochs = 10
path = 'ag_news_csv/'
split = 2900
embedding_size = 300

input, labels = helper.get_data(path)
vocab, vocab_size, word2idx = helper.create_vocab_set()

input, labels = shuffle(input, labels)

encoded_text = helper.get_encoded_text(input, word2idx, sent_limit=sent_limit)

train_X, train_Y, test_X, test_Y = encoded_text[:split], labels[:split], encoded_text[:-split], labels[:-split]

train = helper.batch_gen(train_X, train_Y, batch_size=128)
test = helper.batch_gen(test_X, test_Y, batch_size=644)

x = tf.placeholder(dtype=tf.float32, shape=[None, sent_limit])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
prob = tf.placeholder(dtype=tf.float32)

with tf.name_scope('embedding_layer'):
    embed_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], -1.0, 1.0), name='weight')

# layer 1
with tf.name_scope('layer1'):
    conv_w = tf.Variable(tf.truncated_normal([7, 7, 1, num_feature_map]), name='weight')
    l1_bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, l1_bias)
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

# layer 2
with tf.name_scope('layer2'):
    conv_w = tf.Variable(tf.truncated_normal([7, 7, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

# layer 3
with tf.name_scope('layer3'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)

# layer 4
with tf.name_scope('layer4'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)

# layer 5
with tf.name_scope('layer5'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)

# layer 6
with tf.name_scope('layer6'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

num_features_total = 34 * num_feature_map
output = tf.reshape(output, [-1, num_features_total], name='reshape')

# layer 7
with tf.name_scope('layer7'):
    w = tf.Variable(tf.truncated_normal([34*num_feature_map, nodes]), name='weight')
    b = tf.Variable(tf.truncated_normal([nodes]), name='bias')

    output = tf.matmul(output, w, name='matmul1') + b
    output = tf.nn.dropout(output, prob)

# layer 8
with tf.name_scope('layer8'):
    w = tf.Variable(tf.truncated_normal([nodes, nodes]), name='weight')
    b = tf.Variable(tf.truncated_normal([nodes]), name='bias')

    output = tf.matmul(output, w, name='matmul2') + b
    output = tf.nn.dropout(output, prob)

# layer 9
with tf.name_scope('output_layer'):
    w = tf.Variable(tf.truncated_normal([nodes, n_classes]), name='weight')
    b = tf.Variable(tf.truncated_normal([n_classes]), name='bias')
    output = tf.matmul(output, w, name='matmul3') + b

with tf.name_scope('loss'):
    logits = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    cost = tf.reduce_mean(logits)

optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('Training on {} inputs with {} classes'.format(train_X.shape[0], len(labels[0])))
print('Testing on {} cases'.format(test_X.shape[0]))

step = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for xx, yy in train:
            print('start')
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={x:xx, y:yy, prob:0.5})
            print('finish')
            print('Batch {}  -  Loss {}  -  Acc {}'.format(step, loss, acc))
            step += 1
    for xx, yy in test:
        acc = sess.run(accuracy, feed_dict={x:xx, y:yy, prob:1.})
        print(acc)