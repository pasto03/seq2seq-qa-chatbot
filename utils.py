from seq2seq import Seq2Seq

from encoder import Encoder
from decoder import Decoder

import torch
import pickle

from word2seq import Word2Seq


# load word dict
ws = pickle.load(open("ws.pkl", "rb"))


# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Model will be using {} device.".format(device))

batch_size = 64
# seq_len = 10
src_vocab_size = len(ws.vocab) + 1
tgt_vocab_size = len(ws.vocab) + 1
embedding_dim = 250  # up: accuracy increase
hidden_size = 1024   # up: accuracy increase significantly
p = 0.1
teacher_force_ratio = 0.5

# params from train dataset, need to be changed if train new dataset
max_len_q = 14
max_len_a = 39


# model conf.
encoder = Encoder(input_size=src_vocab_size, embedding_size=embedding_dim, hidden_size=hidden_size,
                  p=p)
decoder = Decoder(input_size=tgt_vocab_size, embedding_size=embedding_dim, hidden_size=hidden_size,
                    output_size=tgt_vocab_size, p=p)

model = Seq2Seq(encoder, decoder, tgt_vocab_size, teacher_force_ratio=teacher_force_ratio, device=device)


# please change the function by input
def encode_input(q):
    import jieba
    import logging
    jieba.setLogLevel(logging.INFO)
    q = ws.fit(list(jieba.cut(q)), max_len=max_len_q)
    q = torch.tensor(q).unsqueeze(1)

    return q

def decode_output(output):
    return ''.join(ws.reverse_fit(output.tolist(), no_pad=True))