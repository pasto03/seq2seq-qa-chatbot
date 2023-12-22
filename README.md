# Seq2Seq QA Chatbot
A Q&amp;A chatbot built using seq2seq-attention architecture


# Use
Run _chat.py_ to play with the chatbot.

## Dataset Used
Dataset used by model is generated by gpt3.5-16k API, with 20 x 3 question and answer pairs for each language.

## Example
In _chat.py_:
```
...
bot = QAChatbot(language='en')
bot.run(save_log=True)
```

Available language: ['en', 'zh']

## Output
An example using English QA model:
<pre>User: what is nlp
Bot : Natural Language Processing ( NLP ) is a field that focuses on the study of how computers can understand and process human language .

User: tools used in nlp
Bot : In natural language processing , common tools include word to a field that specializes in enabling , language models , and process human language .

User: sentiment analysis
Bot : NLP stands for Natural Language Processing , which involves the study of how computers can understand and process human language .

User: how word embedding workds
Bot : NLP stands for Natural Language Processing , which involves to a vector representation in a high - dimensional space , ensuring that semantically similar words .

User: models that can be used for text generation
Bot : In NLP , common NLP , commonly used for text generation include Recurrent Neural Networks ( RNNs ) , Long Short - Term Memory Networks ( LSTMs ) , and Transformers .</pre>

An example using Chinese QA model:
<pre>User: NLP的挑战有哪些
Bot : 在NLP领域，常用的工具包括词嵌入（word embeddings）、分词器（tokenizer）、语言模型和情感分析器。

User: 词嵌入的实现
Bot : 在NLP中，词嵌入通过将每个单词映射到高维空间中的向量表示，使得语义相近的单词在空间中距离较近，从而更好地捕捉单词之间的语义关系

User: 预训练模型
Bot : 在NLP中，预训练模型如BERTBERT、GPT和RoBERTa等被，取得了自然语言处理的自然语言。

User: 潜在语义分析
Bot : 在NLP中，潜在语义分析通过将文本数据映射到潜在语义空间中，以便提高模型对语义理解的语义。

User: 信息检索方法
Bot : 在NLP中，NLP通过神经网络模型学习语言之间的检索，实现了支持的检索等。
</pre>
