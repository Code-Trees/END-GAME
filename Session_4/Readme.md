# **4 - RNNs & LSTMs**

**Assignment:** 

1. Refer to online remove RNN and add LSTM to the model. 
2. Refer to this . 
   1. The questions this time are already mentioned in the file. Take as much time as you want (but less than 7 days), to solve the file. Once you are done, then write your solutions in the quiz. 
   2. Please note that the **Session 4 Assignment Solution** will time out after 15 minutes, as you just have to copy-paste your answers. 



------



## **LSTM_sentiment_analysis model**

​	We are Replacing the RNN model with LSTM , modifying the code to make it  functional . Please find the RNN model link attached 

[here]: https://github.com/Code-Trees/END-GAME/blob/main/Session_4/RNN_model.ipynb	"RNN code."

- As this is for learning purpose we are first modifying the **tokenizer Spacy**  to a simple by by adding the below code . **What is tokenizer ?** Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters, or sub words. Hence, tokenization can be broadly classified into 3 types – word, character, and sub word (n-gram characters) tokenization.

```python
# just split the data by " "(space) between the words.
def tokenize(s):
    return s.split(' ')
```





