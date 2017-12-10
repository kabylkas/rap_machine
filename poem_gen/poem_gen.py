import pandas as pd, numpy as np, tensorflow as tf
from tensorflow.python.layers import core as layers_core
import lyrics
import dataIterator
import sys

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()
 
def build_graph(vocab_size, state_size, batch_size):
  print("Building graph")
  reset_graph()

  ############################
  #ENCODER
  ############################
  #placeholders
  encoder_inputs = tf.placeholder(tf.int32, [batch_size, None], name="encoder_inputs")
  encoder_seqlen = tf.placeholder(tf.int32, [batch_size], name="encoder_seqlen")
  keep_prob = tf.placeholder_with_default(1.0, []) 

  #embedding layer
  embedding_encoder = tf.get_variable('embedding_encoder_m', [vocab_size, state_size])
  encoder_rnn_inputs = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)

  #rnn
  encoder_cell = tf.nn.rnn_cell.GRUCell(state_size)
  init_state = tf.get_variable('init_state', [1, state_size], initializer=tf.constant_initializer(0.0))
  init_state = tf.tile(init_state, [batch_size, 1])
  encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_rnn_inputs, sequence_length=encoder_seqlen, initial_state=init_state)

  # Add dropout, as the model otherwise quickly overfits
  #encoder_outputs = tf.nn.dropout(encoder_outputs, keep_prob)
  #idx = tf.range(batch_size)*tf.shape(encoder_outputs)[1] + (encoder_seqlen - 1)
  #encoder_state = tf.gather(tf.reshape(encoder_outputs, [-1, state_size]), idx)

  ############################
  #DECODER
  ############################
  #placeholders
  decoder_inputs = tf.placeholder(tf.int32, [None, batch_size]) 
  decoder_outputs = tf.placeholder(tf.int32, [None, batch_size])
  decoder_seqlen = tf.placeholder(tf.int32, [batch_size])

  #embedding layer
  embedding_decoder = tf.get_variable('embedding_decoder_m', [vocab_size, state_size])
  decoder_rnn_inputs = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

  projection_layer = layers_core.Dense(vocab_size, use_bias=False)

  #rnn
  decoder_cell = tf.nn.rnn_cell.GRUCell(state_size)
  helper = tf.contrib.seq2seq.TrainingHelper(decoder_rnn_inputs, decoder_seqlen, time_major=True)
  decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_final_state, output_layer=projection_layer)
  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
  logits = outputs.rnn_output  

  #target_weights = tf.placeholder(tf.float32, [batch_size, None])
  
  #max_time = tf.reduce_max(decoder_seqlen)
  #target_weights = tf.sequence_mask(decoder_seqlen, max_time, dtype=logits.dtype)
  crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
  train_loss = (tf.reduce_sum(crossent)/batch_size)

  # Calculate and clip gradients
  max_gradient_norm = 5
  learning_rate = 0.001
  params = tf.trainable_variables()
  gradients = tf.gradients(train_loss, params)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

  # Optimization
  optimizer = tf.train.AdamOptimizer(learning_rate)
  update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


  ############################
  #INFERENCE
  ############################
  tgt_sos_id = 0
  tgt_eos_id = 1
  #helper
  inf_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([batch_size], tgt_sos_id), tgt_eos_id)
  #decoder
  inf_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, inf_helper, encoder_final_state, output_layer=projection_layer) 
  #dynamic decoding
  source_sequence_length = 10
  maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
  inf_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inf_decoder, maximum_iterations=maximum_iterations)
  generate_lines = tf.identity(inf_outputs.sample_id, name="generate_lines")
  
  return {"train_loss": train_loss, \
          "update_step": update_step,\
          "e_in": encoder_inputs,\
          "e_len": encoder_seqlen,\
          "d_in": decoder_inputs,\
          "d_out": decoder_outputs,\
          "d_len": decoder_seqlen}

def train_graph(graph, iterator, batch_size, num_epochs, model_path):
  print("Starting trainging")
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    current_epoch = 0
    saver = tf.train.Saver()
    while current_epoch<num_epochs:
      current_epoch+=1
      e_batch, e_len, d_batch, d_len = iterator.next_batch(batch_size)
      #put <s> and </s> tokens
      target_weights = np.zeros([batch_size, max(d_len)-1])
      t_d_batch = np.transpose(d_batch)
      d_batch_in = np.transpose(np.delete(t_d_batch, d_batch.shape[1]-1, 0)) 
      d_batch_out = np.transpose(np.delete(t_d_batch, 0, 0)) 
      #some small tweeks before feeding to the graph
      for i in range(batch_size):
        d_len[i]-=1
        if len(d_batch_in[i])!=d_len[i]:
          d_batch_in[i][d_len[i]] = 0
        for j in range(d_len[i]):
          target_weights[i,j] = 1
      #feed to the graph
      feed = {graph["e_in"]:  e_batch,\
              graph["e_len"]: e_len,\
              graph["d_in"]:  np.transpose(d_batch_in),\
              graph["d_out"]: np.transpose(d_batch_out),\
              graph["d_len"]: d_len}

      loss, _ = sess.run([graph["train_loss"], graph["update_step"]], feed_dict = feed)
      if current_epoch%1000 == 0:
        print(loss)

    print("Saving model to {0}".format(model_path))
    saver.save(sess, "{0}poem_gen".format(model_path))

def infer(data, seed, batch_size, num_lines, model_path):
  print("Restoring model from {0}".format(model_path))
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph("{0}poem_gen.meta".format(model_path)) 
    saver.restore(sess,tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()
    encoder_inputs = graph.get_tensor_by_name("encoder_inputs:0")
    encoder_seqlen = graph.get_tensor_by_name("encoder_seqlen:0")
    generate_lines = graph.get_tensor_by_name("generate_lines:0") 
    
    all_lines = seed+"\n"
    line = data.tokenize(seed)
    for k in range(num_lines):
      e_in = np.zeros([batch_size, len(line)])
      for i in range(len(line)):
        e_in[0][i] = line[i]
      e_len = np.ones(batch_size)
      e_len[0] = len(line)
      feed = {encoder_inputs: e_in, encoder_seqlen: e_len} 
      next_line = sess.run([generate_lines], feed_dict=feed)
      next_line = next_line[0][0]
      line = next_line
      all_lines+=data.tokens_to_sentence(next_line)

    print(all_lines)

def process_args(args):
  if len(args)<2:
    print("Usage:")
    print("  For training the model: poem_gen.py -t")
    print("  After training, for inference: poem_gen -i 'seed for the generator'")
    exit(0)
  elif args[1] == "-t":
    return "t", "nothing"
  elif args[1] == "-i":
    return "i", args[2]
  else:
    print("Usage:")
    print("  For training the model: poem_gen.py -t")
    print("  After training, for inference: poem_gen.py -i 'seed for the generator'")
    exit(0)
#############
#MAIN
#############
action, seed = process_args(sys.argv)

encoder_file_path = "./input/2pac.in.encoder"
decoder_file_path = "./input/2pac.in.decoder"
model_path = "./model/"

encoder = lyrics.lyrics(encoder_file_path)
decoder = lyrics.lyrics(decoder_file_path)
iterator = dataIterator.PaddedDataIterator(encoder.df, decoder.df)
batch_size = 10
state_size = 64

if action == "t":
  tf_graph = build_graph(vocab_size=len(encoder.d), \
                         state_size=state_size, \
                         batch_size=batch_size)

  train_graph(graph=tf_graph, \
              iterator=iterator, \
              batch_size=batch_size, \
              num_epochs = 12000,\
              model_path = model_path)
elif action == "i":
  print(seed)
  infer(data=decoder,\
        seed = seed,\
        batch_size = batch_size,\
        num_lines = 4,\
        model_path=model_path)
