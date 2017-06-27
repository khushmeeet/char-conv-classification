import tensorflow as tf
import helper
from sklearn.utils import shuffle
import time

# ----- Params -----
sent_limit = 200
n_classes = 4
num_feature_map = 256
nodes = 1024
epochs = 10
path = 'ag_news_csv/'
embedding_size = 300
batch_size = 128

# ----- Getting the data -----
train_X, train_Y = helper.get_data(path+'train.csv')
test_X, test_Y = helper.get_data(path+'test.csv')

vocab, vocab_size, word2idx = helper.create_vocab_set(train_X)

inputs, labels = shuffle(train_X, train_Y)

encoded_text = helper.get_encoded_text(train_X, word2idx, sent_limit=sent_limit)

# ----- Getting batch generator -----
train = helper.batch_gen(encoded_text, labels, batch_size=batch_size)
test = helper.batch_gen(test_X, test_Y, batch_size=batch_size)

# ----- Tensorflow Graph -----
x = tf.placeholder(dtype=tf.int32, shape=[None, sent_limit], name='inputs')
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='labels')
prob = tf.placeholder(dtype=tf.float32)

with tf.name_scope('embedding_layer'):
    embed_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], -1.0, 1.0), name='weight')
    embedding = tf.nn.embedding_lookup(embed_w, x)
    embedding = tf.expand_dims(embedding, -1)

# ----- layer 1 -----
with tf.name_scope('layer1'):
    conv_w = tf.Variable(tf.truncated_normal([7, 7, 1, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(embedding, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

    tf.summary.histogram('weights', conv_w)
    tf.summary.histogram('bias', bias)

# ----- layer 2 -----
with tf.name_scope('layer2'):
    conv_w = tf.Variable(tf.truncated_normal([7, 7, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

    tf.summary.histogram('weights', conv_w)
    tf.summary.histogram('bias', bias)

# ----- layer 3 -----
with tf.name_scope('layer3'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)

    tf.summary.histogram('weights', conv_w)
    tf.summary.histogram('bias', bias)

# ----- layer 4 ------
with tf.name_scope('layer4'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)

    tf.summary.histogram('weights', conv_w)
    tf.summary.histogram('bias', bias)

# ----- layer 5 -----
with tf.name_scope('layer5'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)

    tf.summary.histogram('weights', conv_w)
    tf.summary.histogram('bias', bias)

# ----- layer 6 -----
with tf.name_scope('layer6'):
    conv_w = tf.Variable(tf.truncated_normal([3, 3, num_feature_map, num_feature_map]), name='weight')
    bias = tf.Variable(tf.truncated_normal([num_feature_map]), name='bias')

    output = tf.nn.conv2d(output, conv_w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')

    tf.summary.histogram('weights', conv_w)
    tf.summary.histogram('bias', bias)

num_features_total = 34 * num_feature_map
output = tf.reshape(output, [-1, num_features_total], name='reshape')

# ----- layer 7 -----
with tf.name_scope('layer7'):
    w = tf.Variable(tf.truncated_normal([34*num_feature_map, nodes]), name='weight')
    b = tf.Variable(tf.truncated_normal([nodes]), name='bias')

    output = tf.matmul(output, w, name='matmul1') + b
    output = tf.nn.dropout(output, prob)

    tf.summary.histogram('weights', w)
    tf.summary.histogram('bias', b)

# ----- layer 8 -----
with tf.name_scope('layer8'):
    w = tf.Variable(tf.truncated_normal([nodes, nodes]), name='weight')
    b = tf.Variable(tf.truncated_normal([nodes]), name='bias')

    output = tf.matmul(output, w, name='matmul2') + b
    output = tf.nn.dropout(output, prob)

    tf.summary.histogram('weights', w)
    tf.summary.histogram('bias', b)

# ----- layer 9 -----
with tf.name_scope('output_layer'):
    w = tf.Variable(tf.truncated_normal([nodes, n_classes]), name='weight')
    b = tf.Variable(tf.truncated_normal([n_classes]), name='bias')
    output = tf.matmul(output, w, name='matmul3') + b

    tf.summary.histogram('weights', w)
    tf.summary.histogram('bias', b)

with tf.name_scope('loss'):
    logits = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    cost = tf.reduce_mean(logits)
    tf.summary.scalar('loss', cost)

optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# ----- summaries -----
merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter('summaries/run_1')


print('Training on {} inputs with {} classes'.format(len(train_X), len(labels[0])))
print('Testing on {} cases'.format(len(test_X)))

step = 0
batches = len(train_X)/batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    for e in range(epochs):
        for xx, yy in train:
            start = time.time()
            loss, _, acc, x = sess.run([cost, optimizer, accuracy, merged_summaries], feed_dict={x: xx, y: yy, prob: 0.5})
            writer.add_summary(x, step)
            end = time.time()
            print('Epoch {} [Step {} / {}]  -  Loss {}  -  Acc {}  -  Took {}'.format(e, step, batches, loss, acc, end-start))
            step += 1
    for xx, yy in test:
        acc = sess.run(accuracy, feed_dict={x: xx, y: yy, prob:1.})
        print(acc)