# First Hands-on

**TO-DO**



1. Add  "Back Translate", i.e. using Google translate to convert the sentences. It has "random_swap" function, as well as "random_delete". 
2. Use "Back Translate", "random_swap" and "random_delete" to augment the data you are training on.
3. Download the StanfordSentimentAnalysis Dataset from this [link (Links to an external site.)](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)(it might be troubling to download it, so force download on chrome). Use "datasetSentences.txt" and "sentiment_labels.txt" files from the zip you just downloaded as your dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.
4. Train your model and achieve **60%+ validation/text accuracy**. Upload your collab file on GitHub with readme that contains details about your assignment/word (minimum **250 words**), **training logs showing final validation accuracy, and outcomes for 10 example inputs from the test/validation data.**
5. **You must submit before DUE date (and not "until" date).**





**1> Adding Data Augmentation Technique.**



https://github.com/Code-Trees/END-GAME/blob/main/Session_5/Augmentation.ipynb

We have the below code to  which can do the data augmentation bu calling the functions:



```python
def back_translate(sequence,lab, PROB = 1):
    languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',
                 'sw', 'vi', 'es', 'el']
    
    #instantiate translator
    translator = Translator()
    
    #store original language so we can convert back
    org_lang = translator.detect(sequence).lang
    
    #randomly choose language to translate sequence to  
    random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])
    #print(random_lang)
    if org_lang in languages:
        #translate to new language and back to original
        translated = translator.translate(sequence, dest = random_lang).text
        #translate back to original language
        translated_back = translator.translate(translated, dest = org_lang).text
        #print(translated,translated_back)
        #apply with certain probability
        if np.random.uniform(0, 1) <= PROB:
            output_sequence = translated_back
        else:
            output_sequence = sequence
            
    #if detected language not in our list of languages, do nothing
    else:
        output_sequence = sequence
    
    return output_sequence,lab


def random_deletion(words,lab, p=0.5): 
    if len(words) == 1: # return if single word
        return words
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words)) 
    
    if len(remaining) == 0: # if not left, sample a random word
        return [random.choice(words)] ,lab
    else:
        return remaining,lab

def random_swap(sentence,lab, n=5): 
    length = range(len(sentence)) 
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1] 
    return sentence,lab
```





**Function to use.**

```python
# train_data.sentence
# train_data.sentiment
df = Train_df.copy()
pbar = tqdm(range(5492+1057,df.shape[0]))
count = len(df)
print (f'Before the shape was :{len(df)}' )

aug_data = []
aug_label = []
for i in pbar:
    
    
    word,val = random_pick(df)
    word1,val1 = back_translate(word,val)
    
    word,val = random_pick(df)
    word = word.split()
    word2,val2 = random_deletion(word,val)
    word2 = ' '.join(i for i in word2)
    
    word,val = random_pick(df)
    word = word.split()
    word3,val3 = random_swap(word,val)
    word3 = ' '.join(i for i in word3)
     
    ins = {'sentence':[word1,word2,word3],'label':[val1,val2,val3]}
    df2 = pd.concat([df2,pd.DataFrame(ins)])
    df2.to_csv('all_transforms_1.csv')
pbar.set_description(desc = f'Loop:{i}')
    
```

Once we have the Aug data ready we are saving it in **all_transforms_1.csv** files.

```python
  0%|          | 0/4205 [00:00<?, ?it/s]
Before the shape was :10754
 30%|██▉       | 1258/4205 [36:24<1:25:02,  1.73s/it]
```





**2> Data Reading.**

```python
def get_phrase_sentiments(base_directory):
    def group_labels(label):
        if label in ["very negative", "negative"]:
            return "negative"
        elif label in ["positive", "very positive"]:
            return "positive"
        else:
            return "neutral"

    dictionary = pd.read_csv(os.path.join(base_directory, "dictionary.txt"), sep="|")
    dictionary.columns = ["phrase", "id"]
    dictionary = dictionary.set_index("id")

    sentiment_labels = pd.read_csv(os.path.join(base_directory, "sentiment_labels.txt"), sep="|")
    sentiment_labels.columns = ["id", "sentiment"]
    sentiment_labels = sentiment_labels.set_index("id")

    phrase_sentiments = dictionary.join(sentiment_labels)

    phrase_sentiments["fine"] = pd.cut(phrase_sentiments.sentiment, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                           include_lowest=True,
                                           labels=["very negative", "negative", "neutral", "positive", "very positive"])
    phrase_sentiments["coarse"] = phrase_sentiments.fine.apply(group_labels)
    return phrase_sentiments


def get_sentence_partitions(base_directory):
    sentences = pd.read_csv(os.path.join(base_directory, "datasetSentences.txt"), index_col="sentence_index",
                                sep="\t")
    splits = pd.read_csv(os.path.join(base_directory, "datasetSplit.txt"), index_col="sentence_index")
    return sentences.join(splits)


def partition(base_directory):
    phrase_sentiments = get_phrase_sentiments(base_directory).reset_index(level=0)
    sentence_partitions = get_sentence_partitions(base_directory)
    # noinspection PyUnresolvedReferences
    data = sentence_partitions.join(phrase_sentiments.set_index("phrase"), on="sentence")
    data["splitset_label"] = data["splitset_label"].fillna(1).astype(int)
    # data["sentence"] = data["sentence"].str.replace(r"\s('s|'d|'re|'ll|'m|'ve|n't)\b", lambda m: m.group(1))
    return data.groupby("splitset_label")
```



We are converting The sentiments into 5 buckets by using  this logic.

```python
def discretize_label(label):
    if label <= 0.05*100: return 'Class1'
    if label <= 0.1*100: return 'Class2'
    if label <= 0.15*100: return 'Class3'
    if label <= 0.2*100: return 'Class4'
    return 'Class5'
```



**2>Model Building ** 



```python
# Model class
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, input_dim, output_dim,emb_dim, hidden_dim, n_layers, dropout):
        # input_dim <--- vocabulary size
        # output_dim <--- len ([positive, negative]) == 2 
        # emb_dim <--- embedding dimension of embedding matrix

        super(Model, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout,batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src,Len):
        # shape: [source_len, batch_size]
        embedded = self.dropout(self.embedding(src)) # shape: [src_len, batch_size, embed_dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, Len.to('cpu'),batch_first=True)
        output, (hidden, cell) = self.encoder(packed) 
        # output shape -> [batch, hidden_dim]
        # hiddden shape -> [n_layers, batch, hidden_dim]
        # cell shape -> [n_layers, batch, hidden_dim]
        output = F.relu(self.fc1(hidden))
#         output = self.dropout(output)
        output = self.fc2(output)
        output = F.softmax(output, dim=1)
#         output = F.relu(output)
        
        return output[-1].squeeze(0)
```

```python
Epoch 0  Train data  Batch No : 697 Epoch Loss: 1131.506 Loss : 1.620 Accuracy : 61.27% : 100%|██████████| 698/698 [01:16<00:00,  9.17it/s]
Epoch 0  TEST DATA Batch No : 34 Epoch Loss: 56.860  Loss : 1.723 | Accuracy : 27.22%: 100%|██████████| 35/35 [00:00<00:00, 44.86it/s]
Epoch 1  Train data  Batch No : 697 Epoch Loss: 1134.099 Loss : 1.653 Accuracy : 60.38% : 100%|██████████| 698/698 [01:15<00:00,  9.20it/s]
Epoch 1  TEST DATA Batch No : 34 Epoch Loss: 56.804  Loss : 1.559 | Accuracy : 26.81%: 100%|██████████| 35/35 [00:00<00:00, 44.71it/s]
Epoch 2  Train data  Batch No : 697 Epoch Loss: 1131.731 Loss : 1.620 Accuracy : 60.94% : 100%|██████████| 698/698 [01:15<00:00,  9.19it/s]
Epoch 2  TEST DATA Batch No : 34 Epoch Loss: 56.419  Loss : 1.559 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 44.20it/s]
Epoch 3  Train data  Batch No : 697 Epoch Loss: 1132.453 Loss : 1.620 Accuracy : 60.61% : 100%|██████████| 698/698 [01:15<00:00,  9.19it/s]
Epoch 3  TEST DATA Batch No : 12 Epoch Loss: 21.185  Loss : 1.622 | Accuracy : 9.29%:  26%|██▌       | 9/35 [00:00<00:00, 79.05it/s]
Epoch     4: reducing learning rate of group 0 to 3.4492e-01.
Epoch 3  TEST DATA Batch No : 34 Epoch Loss: 57.018  Loss : 1.641 | Accuracy : 27.43%: 100%|██████████| 35/35 [00:00<00:00, 43.99it/s]
Epoch 4  Train data  Batch No : 697 Epoch Loss: 1132.687 Loss : 1.620 Accuracy : 60.40% : 100%|██████████| 698/698 [01:15<00:00,  9.20it/s]
Epoch 4  TEST DATA Batch No : 34 Epoch Loss: 57.051  Loss : 1.718 | Accuracy : 26.46%: 100%|██████████| 35/35 [00:00<00:00, 44.36it/s]
Epoch 5  Train data  Batch No : 697 Epoch Loss: 1130.763 Loss : 1.622 Accuracy : 60.57% : 100%|██████████| 698/698 [01:16<00:00,  9.10it/s]
Epoch 5  TEST DATA Batch No : 34 Epoch Loss: 56.542  Loss : 1.636 | Accuracy : 27.66%: 100%|██████████| 35/35 [00:00<00:00, 41.50it/s]
Epoch 6  Train data  Batch No : 697 Epoch Loss: 1131.920 Loss : 1.653 Accuracy : 60.99% : 100%|██████████| 698/698 [01:21<00:00,  8.56it/s]
Epoch 6  TEST DATA Batch No : 34 Epoch Loss: 56.808  Loss : 1.723 | Accuracy : 27.31%: 100%|██████████| 35/35 [00:00<00:00, 41.13it/s]
Epoch 7  Train data  Batch No : 697 Epoch Loss: 1131.405 Loss : 1.650 Accuracy : 61.00% : 100%|██████████| 698/698 [01:21<00:00,  8.52it/s]
Epoch 7  TEST DATA Batch No : 34 Epoch Loss: 56.865  Loss : 1.713 | Accuracy : 27.71%: 100%|██████████| 35/35 [00:00<00:00, 41.33it/s]
Epoch 8  Train data  Batch No : 697 Epoch Loss: 1130.847 Loss : 1.619 Accuracy : 61.31% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 8  TEST DATA Batch No : 11 Epoch Loss: 19.512  Loss : 1.622 | Accuracy : 8.12%:  23%|██▎       | 8/35 [00:00<00:00, 74.04it/s]
Epoch     9: reducing learning rate of group 0 to 5.9485e-02.
Epoch 8  TEST DATA Batch No : 34 Epoch Loss: 56.865  Loss : 1.713 | Accuracy : 27.71%: 100%|██████████| 35/35 [00:00<00:00, 40.78it/s]
Epoch 9  Train data  Batch No : 697 Epoch Loss: 1130.818 Loss : 1.656 Accuracy : 61.40% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 9  TEST DATA Batch No : 34 Epoch Loss: 56.816  Loss : 1.699 | Accuracy : 28.10%: 100%|██████████| 35/35 [00:00<00:00, 40.85it/s]
Epoch 10  Train data  Batch No : 697 Epoch Loss: 1130.422 Loss : 1.653 Accuracy : 61.55% : 100%|██████████| 698/698 [01:21<00:00,  8.52it/s]
Epoch 10  TEST DATA Batch No : 34 Epoch Loss: 56.590  Loss : 1.632 | Accuracy : 28.24%: 100%|██████████| 35/35 [00:00<00:00, 40.86it/s]
Epoch 11  Train data  Batch No : 697 Epoch Loss: 1129.897 Loss : 1.617 Accuracy : 62.68% : 100%|██████████| 698/698 [01:22<00:00,  8.48it/s]
Epoch 11  TEST DATA Batch No : 34 Epoch Loss: 56.925  Loss : 1.709 | Accuracy : 27.62%: 100%|██████████| 35/35 [00:00<00:00, 40.94it/s]
Epoch 12  Train data  Batch No : 697 Epoch Loss: 1129.700 Loss : 1.619 Accuracy : 62.80% : 100%|██████████| 698/698 [01:22<00:00,  8.47it/s]
Epoch 12  TEST DATA Batch No : 34 Epoch Loss: 56.811  Loss : 1.709 | Accuracy : 27.97%: 100%|██████████| 35/35 [00:00<00:00, 41.43it/s]
Epoch 13  Train data  Batch No : 697 Epoch Loss: 1128.914 Loss : 1.614 Accuracy : 62.99% : 100%|██████████| 698/698 [01:22<00:00,  8.47it/s]
Epoch 13  TEST DATA Batch No : 34 Epoch Loss: 56.811  Loss : 1.709 | Accuracy : 27.97%: 100%|██████████| 35/35 [00:00<00:00, 40.89it/s]
Epoch 14  Train data  Batch No : 697 Epoch Loss: 1128.711 Loss : 1.619 Accuracy : 62.63% : 100%|██████████| 698/698 [01:22<00:00,  8.47it/s]
Epoch 14  TEST DATA Batch No : 34 Epoch Loss: 56.977  Loss : 1.709 | Accuracy : 27.71%: 100%|██████████| 35/35 [00:00<00:00, 40.89it/s]
Epoch 15  Train data  Batch No : 697 Epoch Loss: 1129.999 Loss : 1.622 Accuracy : 61.87% : 100%|██████████| 698/698 [01:22<00:00,  8.47it/s]
Epoch 15  TEST DATA Batch No : 34 Epoch Loss: 56.739  Loss : 1.709 | Accuracy : 28.55%: 100%|██████████| 35/35 [00:00<00:00, 41.46it/s]
Epoch 16  Train data  Batch No : 697 Epoch Loss: 1131.515 Loss : 1.622 Accuracy : 61.10% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 16  TEST DATA Batch No : 34 Epoch Loss: 56.828  Loss : 1.559 | Accuracy : 27.35%: 100%|██████████| 35/35 [00:00<00:00, 41.15it/s]
Epoch 17  Train data  Batch No : 697 Epoch Loss: 1130.526 Loss : 1.588 Accuracy : 61.28% : 100%|██████████| 698/698 [01:22<00:00,  8.49it/s]
Epoch 17  TEST DATA Batch No : 11 Epoch Loss: 19.454  Loss : 1.560 | Accuracy : 8.12%:  23%|██▎       | 8/35 [00:00<00:00, 75.17it/s]
Epoch    18: reducing learning rate of group 0 to 1.0259e-02.
Epoch 17  TEST DATA Batch No : 34 Epoch Loss: 56.749  Loss : 1.641 | Accuracy : 28.28%: 100%|██████████| 35/35 [00:00<00:00, 40.74it/s]
Epoch 18  Train data  Batch No : 697 Epoch Loss: 1130.624 Loss : 1.591 Accuracy : 61.34% : 100%|██████████| 698/698 [01:21<00:00,  8.53it/s]
Epoch 18  TEST DATA Batch No : 34 Epoch Loss: 56.570  Loss : 1.632 | Accuracy : 28.51%: 100%|██████████| 35/35 [00:00<00:00, 41.33it/s]
Epoch 19  Train data  Batch No : 697 Epoch Loss: 1131.565 Loss : 1.622 Accuracy : 60.91% : 100%|██████████| 698/698 [01:21<00:00,  8.56it/s]
Epoch 19  TEST DATA Batch No : 34 Epoch Loss: 56.570  Loss : 1.632 | Accuracy : 28.51%: 100%|██████████| 35/35 [00:00<00:00, 40.78it/s]
Epoch 20  Train data  Batch No : 697 Epoch Loss: 1130.736 Loss : 1.624 Accuracy : 61.03% : 100%|██████████| 698/698 [01:21<00:00,  8.56it/s]
Epoch 20  TEST DATA Batch No : 12 Epoch Loss: 20.975  Loss : 1.619 | Accuracy : 9.91%:  23%|██▎       | 8/35 [00:00<00:00, 78.20it/s]
Epoch    21: reducing learning rate of group 0 to 1.7692e-03.
Epoch 20  TEST DATA Batch No : 34 Epoch Loss: 56.639  Loss : 1.564 | Accuracy : 28.06%: 100%|██████████| 35/35 [00:00<00:00, 41.88it/s]
Epoch 21  Train data  Batch No : 697 Epoch Loss: 1131.529 Loss : 1.624 Accuracy : 60.87% : 100%|██████████| 698/698 [01:21<00:00,  8.52it/s]
Epoch 21  TEST DATA Batch No : 34 Epoch Loss: 56.670  Loss : 1.564 | Accuracy : 28.06%: 100%|██████████| 35/35 [00:00<00:00, 40.67it/s]
Epoch 22  Train data  Batch No : 697 Epoch Loss: 1131.941 Loss : 1.653 Accuracy : 60.87% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 22  TEST DATA Batch No : 34 Epoch Loss: 56.612  Loss : 1.564 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 40.56it/s]
Epoch 23  Train data  Batch No : 697 Epoch Loss: 1130.162 Loss : 1.622 Accuracy : 61.39% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 23  TEST DATA Batch No : 12 Epoch Loss: 21.063  Loss : 1.653 | Accuracy : 9.55%:  23%|██▎       | 8/35 [00:00<00:00, 78.97it/s]
Epoch    24: reducing learning rate of group 0 to 3.0512e-04.
Epoch 23  TEST DATA Batch No : 34 Epoch Loss: 56.732  Loss : 1.632 | Accuracy : 27.58%: 100%|██████████| 35/35 [00:00<00:00, 41.77it/s]
Epoch 24  Train data  Batch No : 697 Epoch Loss: 1131.614 Loss : 1.622 Accuracy : 61.12% : 100%|██████████| 698/698 [01:21<00:00,  8.54it/s]
Epoch 24  TEST DATA Batch No : 34 Epoch Loss: 56.773  Loss : 1.632 | Accuracy : 27.58%: 100%|██████████| 35/35 [00:00<00:00, 41.98it/s]
Epoch 25  Train data  Batch No : 697 Epoch Loss: 1130.594 Loss : 1.591 Accuracy : 61.35% : 100%|██████████| 698/698 [01:21<00:00,  8.53it/s]
Epoch 25  TEST DATA Batch No : 34 Epoch Loss: 56.773  Loss : 1.632 | Accuracy : 27.49%: 100%|██████████| 35/35 [00:00<00:00, 40.77it/s]
Epoch 26  Train data  Batch No : 697 Epoch Loss: 1131.106 Loss : 1.622 Accuracy : 61.11% : 100%|██████████| 698/698 [01:22<00:00,  8.49it/s]
Epoch 26  TEST DATA Batch No : 11 Epoch Loss: 19.410  Loss : 1.619 | Accuracy : 8.04%:  23%|██▎       | 8/35 [00:00<00:00, 75.65it/s]
Epoch    27: reducing learning rate of group 0 to 5.2621e-05.
Epoch 26  TEST DATA Batch No : 34 Epoch Loss: 56.773  Loss : 1.632 | Accuracy : 27.49%: 100%|██████████| 35/35 [00:00<00:00, 41.48it/s]
Epoch 27  Train data  Batch No : 697 Epoch Loss: 1131.067 Loss : 1.648 Accuracy : 61.32% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 27  TEST DATA Batch No : 34 Epoch Loss: 56.743  Loss : 1.632 | Accuracy : 27.66%: 100%|██████████| 35/35 [00:00<00:00, 41.05it/s]
Epoch 28  Train data  Batch No : 697 Epoch Loss: 1131.627 Loss : 1.622 Accuracy : 61.35% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 28  TEST DATA Batch No : 34 Epoch Loss: 56.743  Loss : 1.632 | Accuracy : 27.66%: 100%|██████████| 35/35 [00:00<00:00, 41.00it/s]
Epoch 29  Train data  Batch No : 697 Epoch Loss: 1130.216 Loss : 1.622 Accuracy : 61.45% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 29  TEST DATA Batch No : 11 Epoch Loss: 19.382  Loss : 1.622 | Accuracy : 8.12%:  23%|██▎       | 8/35 [00:00<00:00, 77.44it/s]
Epoch    30: reducing learning rate of group 0 to 9.0750e-06.
Epoch 29  TEST DATA Batch No : 34 Epoch Loss: 56.743  Loss : 1.632 | Accuracy : 27.66%: 100%|██████████| 35/35 [00:00<00:00, 41.08it/s]
Epoch 30  Train data  Batch No : 697 Epoch Loss: 1130.184 Loss : 1.624 Accuracy : 61.42% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 30  TEST DATA Batch No : 34 Epoch Loss: 56.743  Loss : 1.632 | Accuracy : 27.58%: 100%|██████████| 35/35 [00:00<00:00, 41.26it/s]
Epoch 31  Train data  Batch No : 697 Epoch Loss: 1131.204 Loss : 1.591 Accuracy : 61.16% : 100%|██████████| 698/698 [01:22<00:00,  8.49it/s]
Epoch 31  TEST DATA Batch No : 34 Epoch Loss: 56.743  Loss : 1.632 | Accuracy : 27.58%: 100%|██████████| 35/35 [00:00<00:00, 41.06it/s]
Epoch 32  Train data  Batch No : 697 Epoch Loss: 1130.684 Loss : 1.591 Accuracy : 61.33% : 100%|██████████| 698/698 [01:22<00:00,  8.48it/s]
Epoch 32  TEST DATA Batch No : 12 Epoch Loss: 21.032  Loss : 1.650 | Accuracy : 9.64%:  23%|██▎       | 8/35 [00:00<00:00, 77.85it/s]
Epoch    33: reducing learning rate of group 0 to 1.5651e-06.
Epoch 32  TEST DATA Batch No : 34 Epoch Loss: 56.700  Loss : 1.632 | Accuracy : 27.93%: 100%|██████████| 35/35 [00:00<00:00, 41.55it/s]
Epoch 33  Train data  Batch No : 697 Epoch Loss: 1131.249 Loss : 1.619 Accuracy : 61.07% : 100%|██████████| 698/698 [01:22<00:00,  8.49it/s]
Epoch 33  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 28.02%: 100%|██████████| 35/35 [00:00<00:00, 41.14it/s]
Epoch 34  Train data  Batch No : 697 Epoch Loss: 1131.103 Loss : 1.653 Accuracy : 61.32% : 100%|██████████| 698/698 [01:22<00:00,  8.48it/s]
Epoch 34  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 28.02%: 100%|██████████| 35/35 [00:00<00:00, 41.55it/s]
Epoch 35  Train data  Batch No : 697 Epoch Loss: 1131.010 Loss : 1.622 Accuracy : 61.13% : 100%|██████████| 698/698 [01:22<00:00,  8.48it/s]
Epoch 35  TEST DATA Batch No : 11 Epoch Loss: 19.382  Loss : 1.622 | Accuracy : 8.12%:  23%|██▎       | 8/35 [00:00<00:00, 76.36it/s]
Epoch    36: reducing learning rate of group 0 to 2.6991e-07.
Epoch 35  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 28.02%: 100%|██████████| 35/35 [00:00<00:00, 40.99it/s]
Epoch 36  Train data  Batch No : 697 Epoch Loss: 1129.915 Loss : 1.622 Accuracy : 61.44% : 100%|██████████| 698/698 [01:22<00:00,  8.47it/s]
Epoch 36  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 28.02%: 100%|██████████| 35/35 [00:00<00:00, 40.76it/s]
Epoch 37  Train data  Batch No : 697 Epoch Loss: 1130.176 Loss : 1.619 Accuracy : 61.24% : 100%|██████████| 698/698 [01:22<00:00,  8.48it/s]
Epoch 37  TEST DATA Batch No : 34 Epoch Loss: 56.732  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 41.91it/s]
Epoch 38  Train data  Batch No : 697 Epoch Loss: 1130.883 Loss : 1.591 Accuracy : 61.17% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 38  TEST DATA Batch No : 12 Epoch Loss: 21.032  Loss : 1.650 | Accuracy : 9.64%:  23%|██▎       | 8/35 [00:00<00:00, 78.04it/s]
Epoch    39: reducing learning rate of group 0 to 4.6549e-08.
Epoch 38  TEST DATA Batch No : 34 Epoch Loss: 56.732  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 41.04it/s]
Epoch 39  Train data  Batch No : 697 Epoch Loss: 1130.264 Loss : 1.622 Accuracy : 61.32% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 39  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 41.05it/s]
Epoch 40  Train data  Batch No : 697 Epoch Loss: 1130.325 Loss : 1.591 Accuracy : 61.58% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 40  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 41.87it/s]
Epoch 41  Train data  Batch No : 697 Epoch Loss: 1130.227 Loss : 1.622 Accuracy : 61.32% : 100%|██████████| 698/698 [01:21<00:00,  8.51it/s]
Epoch 41  TEST DATA Batch No : 12 Epoch Loss: 21.032  Loss : 1.650 | Accuracy : 9.64%:  23%|██▎       | 8/35 [00:00<00:00, 74.24it/s]
Epoch    42: reducing learning rate of group 0 to 8.0279e-09.
Epoch 41  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 41.55it/s]
Epoch 42  Train data  Batch No : 697 Epoch Loss: 1130.479 Loss : 1.622 Accuracy : 61.60% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 42  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 41.04it/s]
Epoch 43  Train data  Batch No : 697 Epoch Loss: 1130.715 Loss : 1.620 Accuracy : 61.32% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 43  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 41.10it/s]
Epoch 44  Train data  Batch No : 697 Epoch Loss: 1131.296 Loss : 1.619 Accuracy : 61.36% : 100%|██████████| 698/698 [01:22<00:00,  8.51it/s]
Epoch 44  TEST DATA Batch No : 34 Epoch Loss: 56.702  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 40.96it/s]
Epoch 45  Train data  Batch No : 697 Epoch Loss: 1129.729 Loss : 1.624 Accuracy : 61.45% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 45  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 40.75it/s]
Epoch 46  Train data  Batch No : 697 Epoch Loss: 1130.616 Loss : 1.622 Accuracy : 61.37% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 46  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 40.97it/s]
Epoch 47  Train data  Batch No : 697 Epoch Loss: 1130.370 Loss : 1.588 Accuracy : 61.09% : 100%|██████████| 698/698 [01:21<00:00,  8.52it/s]
Epoch 47  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 40.71it/s]
Epoch 48  Train data  Batch No : 697 Epoch Loss: 1130.848 Loss : 1.619 Accuracy : 61.14% : 100%|██████████| 698/698 [01:22<00:00,  8.49it/s]
Epoch 48  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 40.55it/s]
Epoch 49  Train data  Batch No : 697 Epoch Loss: 1131.343 Loss : 1.622 Accuracy : 61.23% : 100%|██████████| 698/698 [01:22<00:00,  8.50it/s]
Epoch 49  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 41.24it/s]
Epoch 50  Train data  Batch No : 697 Epoch Loss: 1130.372 Loss : 1.622 Accuracy : 61.11% : 100%|██████████| 698/698 [01:17<00:00,  8.99it/s]
Epoch 50  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 44.24it/s]
Epoch 51  Train data  Batch No : 697 Epoch Loss: 1130.170 Loss : 1.591 Accuracy : 61.28% : 100%|██████████| 698/698 [01:14<00:00,  9.31it/s]
Epoch 51  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 45.37it/s]
Epoch 52  Train data  Batch No : 697 Epoch Loss: 1129.409 Loss : 1.619 Accuracy : 61.42% : 100%|██████████| 698/698 [01:14<00:00,  9.31it/s]
Epoch 52  TEST DATA Batch No : 34 Epoch Loss: 56.730  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 45.06it/s]
Epoch 53  Train data  Batch No : 697 Epoch Loss: 1130.450 Loss : 1.624 Accuracy : 61.16% : 100%|██████████| 698/698 [01:14<00:00,  9.32it/s]
Epoch 53  TEST DATA Batch No : 34 Epoch Loss: 56.727  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 45.31it/s]
Epoch 54  Train data  Batch No : 697 Epoch Loss: 1129.570 Loss : 1.622 Accuracy : 61.41% : 100%|██████████| 698/698 [01:14<00:00,  9.31it/s]
Epoch 54  TEST DATA Batch No : 34 Epoch Loss: 56.727  Loss : 1.632 | Accuracy : 27.75%: 100%|██████████| 35/35 [00:00<00:00, 45.99it/s]
Epoch 55  Train data  Batch No : 697 Epoch Loss: 1131.941 Loss : 1.622 Accuracy : 61.12% : 100%|██████████| 698/698 [01:15<00:00,  9.30it/s]
Epoch 55  TEST DATA Batch No : 34 Epoch Loss: 56.724  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 45.11it/s]
Epoch 56  Train data  Batch No : 697 Epoch Loss: 1132.372 Loss : 1.622 Accuracy : 61.00% : 100%|██████████| 698/698 [01:14<00:00,  9.35it/s]
Epoch 56  TEST DATA Batch No : 34 Epoch Loss: 56.724  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 45.45it/s]
Epoch 57  Train data  Batch No : 697 Epoch Loss: 1130.483 Loss : 1.619 Accuracy : 61.19% : 100%|██████████| 698/698 [01:15<00:00,  9.30it/s]
Epoch 57  TEST DATA Batch No : 34 Epoch Loss: 56.724  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 45.55it/s]
Epoch 58  Train data  Batch No : 697 Epoch Loss: 1131.535 Loss : 1.650 Accuracy : 61.34% : 100%|██████████| 698/698 [01:14<00:00,  9.31it/s]
Epoch 58  TEST DATA Batch No : 34 Epoch Loss: 56.724  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 45.01it/s]
Epoch 59  Train data  Batch No : 697 Epoch Loss: 1129.804 Loss : 1.376 Accuracy : 61.44% : 100%|██████████| 698/698 [01:18<00:00,  8.94it/s]
Epoch 59  TEST DATA Batch No : 34 Epoch Loss: 56.727  Loss : 1.632 | Accuracy : 27.84%: 100%|██████████| 35/35 [00:00<00:00, 44.94it/s]
```



**4>Augmented Data Visualization**

[62]:

```python
ORIGINAL
"The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
```

[62]:

```python
BACK Translation
"The Rock is destined to be the new `` Conan '' of the 21st century and that it will cause an even greater sensation than Arnold Schwarzenegger, Jean-Claud Van Damme or Steven Segal."

```

[62]:

```python
Random deletion
"Rock is 21st '' that going to make a even than Arnold Steven ."
```

[62]:

```python
Random pick
"Century Rock the destined to be '' 21st The 's new `` Conan is and that he 's going to make a splash Steven greater than Arnold Schwarzenegger , Jean-Claud Van Segal or even Damme ."
```
