from seq2seq import Seq2Seq

from encoder import Encoder
from decoder import Decoder

import torch
import pickle

from word2seq import Word2Seq


class QAChatbot:
    def __init__(self, language='zh'):
        self.language = language
        # hyperparameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Model will be using {} device.".format(self.device))

        self.batch_size = 64
        self.embedding_dim = 250  # up: accuracy increase
        self.hidden_size = 1024   # up: accuracy increase significantly
        self.p = 0.1
        self.teacher_force_ratio = 0.0
        
        if language == 'zh':
            # max_len for Chinese dataset
            self.max_len_q = 14
            self.max_len_a = 39
            
            self.ws = ws_zh = pickle.load(open("ws_zh.pkl", "rb"))
            self.PATH = "QA checkpoint/model-20 QA Chinese.pt"
            self.encode_input = self.encode_input_zh
        
        elif language == 'en':
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            
            # max_len for English dataset
            self.max_len_q = 14
            self.max_len_a = 43
            
            self.ws = ws_en = pickle.load(open("ws_en.pkl", "rb"))
            self.PATH = "QA checkpoint/model-20 QA English.pt"
            self.encode_input = self.encode_input_en

        self.src_vocab_size = len(self.ws.vocab) + 1
        self.tgt_vocab_size = len(self.ws.vocab) + 1
    
        # model conf.
        self.encoder = Encoder(input_size=self.src_vocab_size, embedding_size=self.embedding_dim, 
                               hidden_size=self.hidden_size, p=self.p)
        self.decoder = Decoder(input_size=self.tgt_vocab_size, embedding_size=self.embedding_dim, 
                               hidden_size=self.hidden_size, output_size=self.tgt_vocab_size, p=self.p)

        self.model = Seq2Seq(self.encoder, self.decoder, self.tgt_vocab_size, 
                             teacher_force_ratio=self.teacher_force_ratio, device=self.device)
        self.model.load_state_dict(torch.load(self.PATH))
        print("--checkpoint loaded--")
        
    def encode_input_zh(self, q):
        import jieba
        import logging
        jieba.setLogLevel(logging.INFO)
        q = self.ws.fit(list(jieba.cut(q)), max_len=self.max_len_q)
        q = torch.tensor(q).unsqueeze(1)

        return q

    def encode_input_en(self, q: str):
        q = self.ws.fit(self.tokenize([q])[0], max_len=self.max_len_q)
        q = torch.tensor(q).unsqueeze(1)

        return q
    
    def tokenize(self, data):
        return [[token.text for token in doc] for doc in list(self.nlp.pipe(data))]

    def decode_output(self, output):
        space_str = '' if self.language == 'zh' else ' '
        return space_str.join(self.ws.reverse_fit(output.tolist(), no_pad=True))
    
    def run(self, save_log=False):
        # disable teacher force to check actual ability
        self.model.teacher_force_ratio = 0.
        self.model.eval()

        if save_log:
            f = open("chatlog.txt", "wb")
        while True:
            with torch.no_grad():
                question = input("User: ")
                if question == "quit" or question == "q":
                    break
                q_t = self.encode_input(question).to(self.device)
                tgt_empty = torch.zeros([self.max_len_a, 1], dtype=torch.long).to(self.device)

                out = self.model(q_t.to(self.device), tgt_empty)
                out = out.argmax(dim=2).flatten()[1:]

                out_text = self.decode_output(out)
                print("Bot: " + out_text, end='\n\n')

                if save_log:
                    f.write(("User: "+ question + '\n' + "Bot : " + out_text + '\n\n').encode('utf-8'))


bot = QAChatbot(language='en')
bot.run(save_log=True)
