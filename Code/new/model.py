import gzip
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import gzip
import simplejson
import json
import _pickle as pickle
import nltk

def model_inputs():
    """Create palceholders for inputs to the model"""

    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length


def process_encoding_input(target_data, vocab_to_int, batch_size):
    """Remove the last word id from each batch and concat the <GO> to the begining of each batch"""
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    """Create the encoding layer"""
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length, dtype=tf.float32)

    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)
    return enc_output, enc_state


def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, vocab_size, max_summary_length):
    """Create the training logits"""
    _training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                         sequence_length=summary_length,
                                                         time_major=False)

    _training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        _training_helper,
                                                        initial_state,
                                                        output_layer)

    training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(_training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=max_summary_length)
    return training_decoder_output


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_summary_length, batch_size):
    """Create the inference logits"""
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)

    _inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=max_summary_length)

    return _inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    """Create the decoding cell and attention for the training and inference decoding layers"""
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)

    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     text_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                   attn_mech,
                                                   rnn_size)

    initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    initial_state = initial_state.clone(cell_state=enc_state[0])

    with tf.variable_scope("decode"):
        _training_logits = training_decoding_layer(dec_embed_input,
                                                   summary_length,
                                                   dec_cell,
                                                   initial_state,
                                                   output_layer,
                                                   vocab_size,
                                                   max_summary_length)

    with tf.variable_scope("decode", reuse=True):
        _inference_logits = inference_decoding_layer(embeddings,
                                                     vocab_to_int['<GO>'],
                                                     vocab_to_int['<EOS>'],
                                                     dec_cell,
                                                     initial_state,
                                                     output_layer,
                                                     max_summary_length,
                                                     batch_size)

    return _training_logits, _inference_logits


def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    """Use the previous functions to create the training and inference logits"""

    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    _training_logits, _inference_logits = decoding_layer(dec_embed_input,
                                                         embeddings,
                                                         enc_output,
                                                         enc_state,
                                                         vocab_size,
                                                         text_length,
                                                         summary_length,
                                                         max_summary_length,
                                                         rnn_size,
                                                         vocab_to_int,
                                                         keep_prob,
                                                         batch_size,
                                                         num_layers)

    return _training_logits, _inference_logits


def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for _text in pad_texts_batch:
            pad_texts_lengths.append(len(_text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


def read(file):
    with open(file, 'rb') as fp:
        return pickle.load(fp)


""" ------------------------------------ MAIN PROGRAM ---------------------------------------"""
# Set the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.0005
keep_probability = 0.75

sorted_summaries = read('Sorted_labels')
sorted_texts = read('Sorted_data')
vocab_to_int = read('vocab_to_int')
word_embedding_matrix = read('emb_matrix')
int_to_vocab = read('int_to_vocab')

# Build the graph
train_graph = tf.Graph()

# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int) + 1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size)

    print("\n\n")
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                targets,
                                                masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")


"""    ------   Training the Model    ------   """

# Subset the data for training
start = 0
end = start + 45691
sorted_summaries_short = sorted_summaries
sorted_texts_short = sorted_texts
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:", len(sorted_texts_short[-1]))


# Train the Model
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20  # Check training loss after every 20 batches
stop_early = 0
stop = 10  # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3  # Make 3 update checks per epoch
update_check = (len(sorted_texts_short) // batch_size // per_epoch) - 1

update_loss = 0
batch_loss = 0
summary_update_loss = []  # Record the update losses for saving improvements in the model

checkpoint = "best_model.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    # If we want to continue training a previous session
    #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
    #loader.restore(sess, checkpoint)

    for epoch_i in range(1, epochs + 1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: texts_batch,
                 targets: summaries_batch,
                 lr: learning_rate,
                 summary_length: summaries_lengths,
                 text_length: texts_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(sorted_texts_short) // batch_size,
                              batch_loss / display_step,
                              batch_time * display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss / update_check, 3))
                summary_update_loss.append(update_loss)

                # If the update loss is at a new minimum, save the model
                
                if update_loss <= min(summary_update_loss):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0

        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate

        if stop_early == stop:
            print("Stopping Training.")
            break


#
# def text_to_seq(text):
#     '''Prepare the text for the model'''
#
#     #text = clean_text(text)
#     return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]
#
# clean_texts=read('Cleaned_text')
#
#
# def clean_text(text, remove_stopwords=True):
#     '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
#
#     # Convert words to lower case
#     text = text.lower()
#
#     # Replace contractions with their longer forms
#     if True:
#         text = text.split()
#         new_text = []
#         for word in text:
#             if word in contractions:
#                 new_text.append(contractions[word])
#             else:
#                 new_text.append(word)
#         text = " ".join(new_text)
#
#     # Format words and remove unwanted characters
#     text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\<a href', ' ', text)
#     text = re.sub(r'&amp;', '', text)
#     text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
#     text = re.sub(r'<br />', ' ', text)
#     text = re.sub(r'\'', ' ', text)
#
#     # Optionally, remove stop words
#     if remove_stopwords:
#         text = text.split()
#         stops = set(stopwords.words("english"))
#         text = [w for w in text if not w in stops]
#         text = " ".join(text)
#
#     return text
#
#
# input_sentence = clean_texts[5]
# text = text_to_seq(clean_texts[5])
#
# checkpoint = "./best_model.ckpt"
#
# loaded_graph = tf.Graph()
# with tf.Session(graph=loaded_graph) as sess:
#     # Load saved model
#     loader = tf.train.import_meta_graph(checkpoint + '.meta')
#     loader.restore(sess, checkpoint)
#
#     input_data = loaded_graph.get_tensor_by_name('input:0')
#     logits = loaded_graph.get_tensor_by_name('predictions:0')
#     text_length = loaded_graph.get_tensor_by_name('text_length:0')
#     summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
#     keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
#
#     # Multiply by batch_size to match the model's input parameters
#     answer_logits = sess.run(logits, {input_data: [text] * batch_size,
#                                       summary_length: [np.random.randint(5, 8)],
#                                       text_length: [len(text)] * batch_size,
#                                       keep_prob: 1.0})[0]
#
# # Remove the padding from the tweet
# pad = vocab_to_int["<PAD>"]
#
# print('Original Text:', input_sentence)
#
# print('\nText')
# print('  Word Ids:    {}'.format([i for i in text]))
# print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))
#
# print('\nSummary')
# print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
# print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))
