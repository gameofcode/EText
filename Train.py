#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from TextCNN import TextCNN
from tensorflow.contrib import learn
import json
import model_export

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("val_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("anger_file", "./data/anger.text", "help")
tf.flags.DEFINE_string("disgust_file", "./data/disgust.text", "help")
tf.flags.DEFINE_string("fear_file", "./data/fear.text", "help")
tf.flags.DEFINE_string("happy_file", "./data/happy.text", "help")
tf.flags.DEFINE_string("sad_file", "./data/sad.text", "help")
tf.flags.DEFINE_string("surprise_file", "./data/surprise.text", "help")
tf.flags.DEFINE_string("neutral_file", "./data/neutral.text", "help")
tf.flags.DEFINE_string("vocab_dict_file", "./data/vocab_dict.json", "Vocab dictionary.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model export parameters
tf.flags.DEFINE_string("input_graph_name", "input_graph.pb","Graph input file of the graph to export")
tf.flags.DEFINE_string("output_graph_name", "output_graph.pb","Graph output file of the graph to export")
tf.flags.DEFINE_string("output_node","output/predictions32","The output node of the graph")


FLAGS = tf.flags.FLAGS

# It stores path where model and summaries are saved
out_dir = None


def preprocess():
    # Data Preparation
    # ==================================================

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.anger_file, FLAGS.disgust_file, FLAGS.fear_file,
                                                  FLAGS.happy_file, FLAGS.sad_file, FLAGS.surprise_file,
                                                  FLAGS.neutral_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    # max_document_length = 30
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    print("max_document_length : " + str(max_document_length))
    print(vocab_processor)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print(x_text[14])
    print(x[14])

    # Output directory for models and summaries
    global out_dir
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # serialize vocab dict to use outside Python
    vocab_dict = vocab_processor.vocabulary_._mapping
    vocab_dict["__MAX_DOC_LEN__"] = max_document_length
    with open(FLAGS.vocab_dict_file, 'w') as f:
        f.write(json.dumps(vocab_dict))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/validation/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    val_sample_index = dev_sample_index - int(FLAGS.val_sample_percentage * float(len(y)))
    x_train, x_val, x_dev = x_shuffled[:val_sample_index], x_shuffled[val_sample_index:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_val, y_dev = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Val/Dev split: {:d}/{:d}/{:d}".format(len(y_train), len(y_val), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev, x_val, y_val


def train(x_train, y_train, vocab_processor, x_dev, y_dev, x_val, y_val):
    # Training
    # ==================================================

    print("x_train[:2] : ")
    print(x_train[:2])
    print("y_train[:2] : ")
    print(y_train[:2])

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            # global out_dir
            # timestamp = str(int(time.time()))
            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            # print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Val summaries
            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(out_dir, "summaries", "val")
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("train_step => {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def val_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a val set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, val_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("val_step => {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("dev_step => {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            path = None
            print([n.name for n in tf.get_default_graph().as_graph_def().node])
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    val_step(x_val, y_val, writer=val_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # Evaluate in test set
            print("\nDev Evaluation:")
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")

            # Saving graph
            print("Saving graph...")
            tf.train.write_graph(sess.graph, checkpoint_dir, FLAGS.input_graph_name)

            # exporting graph and model
            print("Freezing model...")
            input_graph_path = os.path.join(checkpoint_dir, FLAGS.input_graph_name)
            output_graph_path = os.path.join(checkpoint_dir, FLAGS.output_graph_name)
            model_export.freeze_model(input_graph_path, output_graph_path, FLAGS.output_node, path)


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, x_val, y_val = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, x_val, y_val)


if __name__ == '__main__':
    tf.app.run()
