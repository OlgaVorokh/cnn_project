# -*- coding: utf-8 -*-

import tensorflow as tf


class TextCNN(object):
    def __init__(
            self,
            input_length,
            transitions_number,
            words_vocab_size, tags_vocab_size, labels_vocab_size, embedding_size,
            input_cnt_words, input_cnt_tags, input_cnt_labels,
            l2_reg_lambda=0.0,
            hidden_layer_size=64,
    ):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, input_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, transitions_number], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.embedding_placeholder = tf.placeholder(
            tf.float32,
            [words_vocab_size, embedding_size],
            name='embedding_placeholder',
        )

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        self.words_input = self.input_x[:, :input_cnt_words]
        input_tags_end = input_cnt_words + input_cnt_tags
        self.tags_input = self.input_x[:, input_cnt_words:input_tags_end]
        self.labels_input = self.input_x[:, input_tags_end:]

        # ==================== WORDS line ======================================================
        # Embedding layer for words
        with tf.device('/cpu:0'), tf.name_scope("embedding_words"):
            self.words_W = tf.Variable(
                tf.constant(0.0, shape=[words_vocab_size, embedding_size]),
                trainable=True,
                name="words_embedding_weights",
            )
            self.words_embedding_init = self.words_W.assign(self.embedding_placeholder)

            self.words_embedded_chars = tf.nn.embedding_lookup(self.words_embedding_init, self.words_input)
            self.words_embedded_chars_line = tf.reshape(
                self.words_embedded_chars,
                [-1, input_cnt_words * embedding_size]
            )

        with tf.name_scope('words_hidden'):
            self.words_hidden_W = tf.Variable(
                tf.truncated_normal([input_cnt_words * embedding_size, hidden_layer_size], stddev=0.1),
                name='words_hidden_weights',
            )
            self.words_vector = tf.matmul(self.words_embedded_chars_line, self.words_hidden_W)
        # ==================== End WORDS line ==================================================

        # ==================== TAGS line ======================================================
        # Embedding layer for tags
        with tf.device('/cpu:0'), tf.name_scope("tags_embedding"):
            self.tags_W = tf.Variable(
                tf.random_uniform([tags_vocab_size, embedding_size], -1.0, 1.0),
                name="tags_embedding_weights",
            )
            self.tags_embedded_chars = tf.nn.embedding_lookup(self.tags_W, self.tags_input)
            self.tags_embedded_chars_line = tf.reshape(
                self.tags_embedded_chars,
                [-1, input_cnt_tags * embedding_size]
            )

        with tf.name_scope('tags_hidden'):
            self.tags_hidden_W = tf.Variable(
                tf.truncated_normal([input_cnt_tags * embedding_size, hidden_layer_size], stddev=0.1),
                name='tags_hidden_weights',
            )
            self.tags_vector = tf.matmul(self.tags_embedded_chars_line, self.tags_hidden_W)
        # ==================== End TAGS line ======================================================

        # ==================== LABELS line ======================================================
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('labels_embedding'):
            self.labels_W = tf.Variable(
                tf.random_uniform([labels_vocab_size, embedding_size], -1.0, 1.0),
                name="labels_W"
            )
            self.labels_embedded_chars = tf.nn.embedding_lookup(self.labels_W, self.labels_input)
            self.labels_embedded_chars_line = tf.reshape(
                self.labels_embedded_chars,
                [-1, input_cnt_labels * embedding_size]
            )

        with tf.name_scope('labels_hidden'):
            self.labels_hidden_W = tf.Variable(
                tf.truncated_normal([input_cnt_labels * embedding_size, hidden_layer_size], stddev=0.1),
                name='labels_hidden_weights',
            )
            self.labels_vector = tf.matmul(self.labels_embedded_chars_line, self.labels_hidden_W)
        # ==================== End LABELS line ======================================================

        self.hidden_b = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]), name='hidden_biases')
        self.vectors_sum = tf.add_n([self.words_vector, self.tags_vector, self.labels_vector])
        self.prom = self.vectors_sum + self.hidden_b
        self.vectors_cube = tf.pow(self.prom, 3)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.vectors_cube, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[hidden_layer_size, transitions_number],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[transitions_number]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
