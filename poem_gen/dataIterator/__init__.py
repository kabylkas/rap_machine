import pandas as pd
import numpy as np
import random

class PaddedDataIterator():
  def __init__(self, df1, df2):
    self.df_encoder = df1
    self.df_decoder = df2
    
    self.size = len(self.df_encoder)
    self.sample_count = {}
    self.epochs = 0

  def next_batch(self, n):
    samples = random.sample(range(self.size), n)
    for sample in samples:
      if sample not in self.sample_count:
        self.sample_count[sample] = 1
      else:
        self.sample_count[sample]+=1

    encoder_res = []
    decoder_res = []
    encoder_max_len = 0
    decoder_max_len = 0
    for sample in samples:
      encoder_res.append((self.df_encoder["string"][sample], \
                         self.df_encoder["as_numbers"][sample], \
                         self.df_encoder["length"][sample]))

      if self.df_encoder["length"][sample]>encoder_max_len:
        encoder_max_len = self.df_encoder["length"][sample]

      decoder_res.append((self.df_decoder["string"][sample], \
                          self.df_decoder["as_numbers"][sample][:], \
                          self.df_decoder["length"][sample]+2))

      decoder_res[len(decoder_res)-1][1].append(1)
      decoder_res[len(decoder_res)-1][1].insert(0,0)
      if decoder_res[len(decoder_res)-1][2]>decoder_max_len:
        decoder_max_len = decoder_res[len(decoder_res)-1][2]

    # Pad sequences with 0s so they are all the same length
    encoder_x = np.zeros([n, encoder_max_len], dtype=np.int32) 
    encoder_len = []
    for i, x_i in enumerate(encoder_x):
      x_i[:encoder_res[i][2]] = encoder_res[i][1]
      encoder_len.append(encoder_res[i][2])

    decoder_x = np.zeros([n, decoder_max_len], dtype=np.int32)
    decoder_len = []
    for i, x_i in enumerate(decoder_x):
      x_i[:decoder_res[i][2]] = decoder_res[i][1]
      decoder_len.append(decoder_res[i][2])

    return encoder_x, encoder_len, decoder_x, decoder_len
