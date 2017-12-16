import pandas as pd, numpy as np, tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.python.layers import core as layers_core
import lyrics
import dataIterator
import sys
import random
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import math

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
  #beam search
  beam_width=10
  decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)
  decoder_beam = tf.contrib.seq2seq.BeamSearchDecoder(\
        cell=decoder_cell,\
        embedding=embedding_decoder,\
        start_tokens=[0],\
        end_token=1,\
        initial_state=decoder_initial_state,\
        beam_width=beam_width,\
        output_layer=projection_layer,\
        length_penalty_weight=0.0)

  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
  logits = outputs.rnn_output  
  
  outputs_beam, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_beam, ...)
  next_lines = tf.identity(tf.transpose(outputs_beam.predicted_ids), name="next_lines")
  

  #target_weights = tf.placeholder(tf.float32, [batch_size, None])
  
  max_time = tf.reduce_max(decoder_seqlen)
  target_weights = tf.transpose(tf.sequence_mask(decoder_seqlen, max_time, dtype=logits.dtype))
  crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
  train_loss = (tf.reduce_sum(crossent*target_weights)/batch_size)

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
          "d_len": decoder_seqlen,\
          "max_time": max_time,\
          "crossent": crossent,\
          "target_weights": target_weights,\
          "next_lines": next_lines}

def train_graph(graph, data, train_iterator, test_iterator, batch_size, num_epochs, model_path):
  print("Starting trainging")
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    current_epoch = 0
    saver = tf.train.Saver()
    test_num=0
    #build refence for the blue scoring
    references_as_num = train_iterator.get_bleu_references()
    references = []
    for ref_as_num in references_as_num:
      references.append(data.tokens_to_sentence(ref_as_num).split(" "))
    while current_epoch<num_epochs:
      current_epoch+=1
      e_batch, e_len, d_batch, d_len = train_iterator.next_batch(batch_size)
      t_d_batch = np.transpose(d_batch)
      d_batch_in = np.transpose(np.delete(t_d_batch, d_batch.shape[1]-1, 0)) 
      d_batch_out = np.transpose(np.delete(t_d_batch, 0, 0)) 
      #some small tweeks before feeding to the graph
      for i in range(batch_size):
        d_len[i]-=1
        if len(d_batch_in[i])!=d_len[i]:
          d_batch_in[i][d_len[i]] = 0
      #feed to the graph
      feed = {graph["e_in"]:  e_batch,\
              graph["e_len"]: e_len,\
              graph["d_in"]:  np.transpose(d_batch_in),\
              graph["d_out"]: np.transpose(d_batch_out),\
              graph["d_len"]: d_len}

      loss, _ = sess.run([graph["train_loss"], graph["update_step"]], feed_dict = feed)
      #test every 1000 iterations
      if current_epoch%1000 == 0:
        print("Test #: {0}".format(test_num))
        test_num+=1
        print("Loss on the last batch: {0}".format(loss))
        score = 0
        for test in range(5):
          e_batch, e_len, d_batch, _ = test_iterator.next_batch(1)
          #feed to the graph
          feed = {graph["e_in"]:  e_batch,\
                  graph["e_len"]: e_len}
          
          next_lines = sess.run([graph["next_lines"]], feed_dict=feed)
          next_line = next_lines[0][random.randint(3, 5)][0]
          candidate = data.tokens_to_sentence(next_line).split(" ")
          candidate_str = data.tokens_to_sentence(next_line)
          score += sentence_bleu(references, candidate)
        #print("Test input:   {0}".format(str.replace(data.tokens_to_sentence(e_batch[0]), "\n", "")))
        #print("Test target:  {0}".format(str.replace(data.tokens_to_sentence(d_batch[0]), "\n", "")))
        print("Model output: {0}".format(str.replace(candidate_str, "\n", "")))
        print("Average BLEU score: {0}".format(score/5))
        print("==============================")

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
    nls = graph.get_tensor_by_name("next_lines:0") 
    
    all_lines = seed+"\n"
    line = data.tokenize(seed)
    for k in range(num_lines):
      e_in = np.zeros([batch_size, len(line)])
      for i in range(len(line)):
        if line[i] == -1:
          break
        else:
          e_in[0][i] = line[i]
      e_len = np.ones(batch_size)
      e_len[0] = len(line)
      feed = {encoder_inputs: e_in, encoder_seqlen: e_len} 
      next_lines = sess.run([nls], feed_dict=feed)
      #next_lines = np.transpose(next_lines)
      next_line = next_lines[0][random.randint(1, 6)][0]
      line = next_line
      all_lines+=data.tokens_to_sentence(next_line)+"\n"
      """
      next_line = next_line[0][0]
      line = next_line
      all_lines+=data.tokens_to_sentence(next_line)+"\n"
      """

    print(all_lines)

def process_args(args):
  if len(args)<2:
    print("Usage:")
    print("  For training the model: poem_gen.py -t")
    print("  After training, for inference: poem_gen -i 'seed for the generator'")
    exit(0)
  elif args[1] == "-t":
    return "t", args[2], 0
  elif args[1] == "-d":
    return "dump", args[2], 0
  elif args[1] == "-i":
    return "i", args[2], args[3]
  else:
    print("Usage:")
    print("  For training the model: poem_gen.py -t")
    print("  After training, for inference: poem_gen.py -i 'seed for the generator'")
    exit(0)

def cosine_similarity(vector1, vector2):
  dot_product = np.dot(vector1, vector2)
  magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
  if not magnitude:
      return 0
  return dot_product/magnitude

def get_best_match(seed, data):
  vectorizer = CountVectorizer()
  transformer = TfidfTransformer(smooth_idf=False)

  corpus = []
  for key, line in data.df["string"].items():
    corpus.append(line)

  vectorizer.fit_transform(corpus)
  seed_counts = vectorizer.transform([data.process(seed)]).toarray()
  seed_counts = np.squeeze(np.asarray(seed_counts))
  seed_tfidf = transformer.fit_transform(seed_counts)

  max_sim = 0
  save_line = ""
  for key, line in data.df["string"].items():
    if "**endsong**" in line:
      continue
    line_counts = vectorizer.transform([line]).toarray()
    line_counts = np.squeeze(np.asarray(line_counts))
    line_tfidf = transformer.fit_transform(line_counts)
    sim = cosine_similarity(line_counts, seed_counts) 
    if max_sim<sim:
      max_sim = sim
      save_line = line

  return save_line
#############
#MAIN
#############
random.seed(datetime.now())

action, in_file, seed = process_args(sys.argv)

encoder_file_path = "./input/{0}.encoder".format(in_file)
decoder_file_path = "./input/{0}.decoder".format(in_file)
model_path = "./model_eminem_256/"

encoder = lyrics.lyrics(encoder_file_path, dict_size=3000)
decoder = lyrics.lyrics(decoder_file_path, dict_size=3000)
print(len(encoder.d))
train_len, test_len = np.floor(len(encoder.df)*0.9), np.floor(len(encoder.df)*0.1)
encoder_train_df, encoder_test_df = encoder.df.ix[:train_len-1], encoder.df.ix[train_len:train_len+test_len]
decoder_train_df, decoder_test_df = decoder.df.ix[:train_len-1], decoder.df.ix[train_len:train_len+test_len]

train_iterator = dataIterator.PaddedDataIterator(encoder_train_df, decoder_train_df)
test_iterator = dataIterator.PaddedDataIterator(encoder_test_df, decoder_test_df)

batch_size = 1
state_size = 256

if action == "t":
  tf_graph = build_graph(vocab_size=len(encoder.d), \
                         state_size=state_size, \
                         batch_size=batch_size)

  train_graph(graph=tf_graph, \
              data = decoder, \
              train_iterator=train_iterator, \
              test_iterator=test_iterator, \
              batch_size=batch_size, \
              num_epochs = 5000,\
              model_path = model_path)

elif action == "i":
  print(seed)
  new_seed = get_best_match(seed, encoder)
  infer(data=decoder,\
        seed = new_seed,\
        batch_size = batch_size,\
        num_lines = 10,\
        model_path=model_path)
elif action == "dump":
  with open("vocab.out", "w") as out_file:
    for word in encoder.d:
      out_file.write("{0}\n".format(word))
  
