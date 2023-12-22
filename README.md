# Seq2Seq QA Chatbot
A Q&amp;A chatbot built using seq2seq-attention architecture


# Use
Run _chat.py_ to play with the chatbot.

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
