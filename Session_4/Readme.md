# **4 - RNNs & LSTMs**

**ToDo:** 

1. Refer to online remove RNN and add LSTM to the model. 
2. Refer to this . 
   1. The questions this time are already mentioned in the file. Take as much time as you want (but less than 7 days), to solve the file. Once you are done, then write your solutions in the quiz. 
   2. Please note that the **Session 4 Assignment Solution** will time out after 15 minutes, as you just have to copy-paste your answers. 



------



## **LSTM_Sentiment_analysis model**

​	We are Replacing the RNN model with LSTM , modifying the code to make it  functional . Please find the RNN model link attached .

https://github.com/Code-Trees/END-GAME/blob/main/Session_4/RNN_model.ipynb

[here]: https://github.com/Code-Trees/END-GAME/blob/main/Session_4/RNN_model.ipynb	"RNN code"

##### **<u>Tokenizer:</u>**  

As this is for learning purpose we are first modifying the **tokenizer Spacy**  to a simple by by adding the below code . **What is tokenizer ?** Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters, or sub words. Hence, tokenization can be broadly classified into 3 types – word, character, and sub word (n-gram characters) tokenization.

```python
# just split the data by " "(space) between the words.
def tokenize(s):
    return s.split(' ')

TEXT = Field(tokenize = tokenize,tokenizer_language = 'en_core_web_sm', lower = True)
LABEL = LabelField(dtype = torch.float)
```



##### **<u>Data prep:</u>** 

Once we have the tokenizer lets Split the data . In our case we have the IMDB rating Dataset. This data set is having  50,000 reviews with target as positive / negative. We will split the data with Train data, Validation_data , Test_data with , 17500,7500,25000 split respectively. 

Once we have the data splits we can build vocab from training data. 

```python
# Build vocabulary for source and target from training data

TEXT.build_vocab(train_data, max_size=25_000)
LABEL.build_vocab(train_data)
```

##### <u>**Model Building :**</u> 

We have the model Code ready from RNN code. Let's modify it with below updates. 

1. Input dim:  The Shape of Vocabulary ,that can be the max length of a sentence. 

2. output_dim : len ([positive, negative]) == 2

3. emb_dim : Dimension of embedding matrix

4. hidden_dim : Num of hidden dimension.

5. n_layer: How many LSTM layers to build

6. dropout: The dropout value 

   ```python
   INPUT_DIM = len(TEXT.vocab)
   OUTPUT_DIM = len(LABEL.vocab)
   EMBEDDING_DIM = 100
   HIDDEN_DIM = 256
   N_LAYERS = 1
   DROPOUT = 0.6
   ```

```python
# Model class
class Model(nn.Module):
    def __init__(self, input_dim, output_dim,emb_dim, hidden_dim, n_layers, dropout):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # shape: [source_len, batch_size]
        embedded = self.dropout(self.embedding(src)) # shape: [src_len, batch_size, embed_dim]
        output, (hidden, cell) = self.rnn(embedded) 
        # output shape -> [batch, hidden_dim]
        # hiddden shape -> [n_layers, batch, hidden_dim]
        # cell shape -> [n_layers, batch, hidden_dim]
        output = self.fc1(output[-1])
#         output = self.fc2(self.relu(output))
        return output
```



We are running the model is Cuda.

```python
def gpu_check(seed_val = 1):
    print('The Seed is set to {}'.format(seed_val))
    if torch.cuda.is_available():
        print('Model will Run on CUDA.')
        print ("Type 'watch nvidia-smi' to monitor GPU\n")
        torch.cuda.manual_seed(seed_val)
        device = 'cuda'
    else:
        torch.manual_seed(seed_val)
        print ('Running in CPU')
        device = 'cpu'
    cuda = torch.cuda.is_available()
    return cuda,seed_val,device
```



```python
In [8]:cuda,SEED,device = gpu_check(seed_val=1234)

The Seed is set to 1234
Model will Run on CUDA.
Type 'watch nvidia-smi' to monitor GPU
```

```shell
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   53C    P0    43W / 250W |      2MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```



##### <u>**Train Test Loop : **</u> 

We are using **Adam** as optimiser, Also **lr_scheduler** to find the best Learning rate. Loss as **CrossEntropyLoss**.

Let's create The train loop , evaluate loop to run test loop and validation loop with below code.

```python
def train(EPOCH,model, iterator, optimizer=optimizer, criterion=criterion, clip=1,):
    model.train()
    epoch_loss = 0
    total_correct = 0
    total_count = 0
    pbar = tqdm(iterator)
    for i, batch in enumerate(pbar):
        src = batch.text.to(device)
        trg = batch.label.to(device)
        trg = trg.long()
        optimizer.zero_grad()
        output = model(src)
        
        total_correct += torch.sum(torch.eq(output.argmax(1), trg))
        total_count+=len(trg)
        
        loss = criterion(output, trg)
        
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_description(desc= f'Epoch {EPOCH} Train data Batch No : {i} Loss : {loss.item():.3f} Accuracy : {total_correct/total_count * 100 :.2f}% ' )
    
    train_accuracy.append(total_correct/total_count)
    mean_loss = epoch_loss / len(iterator)
    train_loss.append(mean_loss)
    
    scheduler.step(mean_loss)
```

```python
def evaluate(EPOCH,model, iterator, criterion,typ_loader):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    pbar  = tqdm(iterator)
    with torch.no_grad():
        
        for i,batch in enumerate(pbar):
            src = batch.text.to(device)
            trg = batch.label.to(device)
            trg = trg.long()
            predictions = model(src)
            
            loss = criterion(predictions, trg)
            
            acc = binary_accuracy(predictions, trg)

            epoch_loss += loss.item()
            epoch_acc += acc
            pbar.set_description(desc= f'Epoch {EPOCH} {typ_loader} Batch No : {i} Loss : {loss.item():.3f} Accuracy : {epoch_acc / len(iterator)* 100 :.2f}% ' )
```

##### <u>**Model Running And o/p:**</u>

to Run the model we run below code:

EPOCHS = 5

```python
total_epoch = 5
for epoch in range(total_epoch):
    result = train(epoch,model=model, iterator=train_iterator)
    evaluate(epoch,model,valid_iterator,criterion,'Valid data')
    evaluate(epoch,model,test_iterator,criterion,'Test data')
```

The O/P of out Mehnat is below.

```python
Epoch 0 Train data Batch No : 546 Loss : 0.701 Accuracy : 49.77% : 100%|██████████| 547/547 [00:37<00:00, 14.55it/s]
Epoch 0 Valid data Batch No : 234 Loss : 0.692 Accuracy : 50.31% : 100%|██████████| 235/235 [00:02<00:00, 86.82it/s]
Epoch 0 Test data Batch No : 781 Loss : 0.678 Accuracy : 37.39% : 100%|██████████| 782/782 [00:09<00:00, 86.53it/s]
Epoch 1 Train data Batch No : 546 Loss : 0.705 Accuracy : 49.93% : 100%|██████████| 547/547 [00:37<00:00, 14.44it/s]
Epoch 1 Valid data Batch No : 234 Loss : 0.686 Accuracy : 49.99% : 100%|██████████| 235/235 [00:02<00:00, 89.57it/s]
Epoch 1 Test data Batch No : 781 Loss : 0.704 Accuracy : 57.17% : 100%|██████████| 782/782 [00:08<00:00, 88.91it/s]
Epoch 2 Train data Batch No : 546 Loss : 0.718 Accuracy : 50.41% : 100%|██████████| 547/547 [00:37<00:00, 14.41it/s]
Epoch 2 Valid data Batch No : 234 Loss : 0.692 Accuracy : 51.52% : 100%|██████████| 235/235 [00:02<00:00, 89.56it/s]
Epoch 2 Test data Batch No : 781 Loss : 0.629 Accuracy : 46.12% : 100%|██████████| 782/782 [00:08<00:00, 89.88it/s]
Epoch 3 Train data Batch No : 546 Loss : 0.679 Accuracy : 50.71% : 100%|██████████| 547/547 [00:37<00:00, 14.61it/s]
Epoch 3 Valid data Batch No : 234 Loss : 0.685 Accuracy : 50.16% : 100%|██████████| 235/235 [00:02<00:00, 89.85it/s]
Epoch 3 Test data Batch No : 781 Loss : 0.741 Accuracy : 57.04% : 100%|██████████| 782/782 [00:08<00:00, 89.43it/s]
Epoch 4 Train data Batch No : 546 Loss : 0.719 Accuracy : 50.35% : 100%|██████████| 547/547 [00:37<00:00, 14.60it/s]
Epoch 4 Valid data Batch No : 234 Loss : 0.711 Accuracy : 49.58% : 100%|██████████| 235/235 [00:02<00:00, 88.88it/s]
Epoch 4 Test data Batch No : 781 Loss : 0.778 Accuracy : 56.24% : 100%|██████████| 782/782 [00:08<00:00, 89.35it/s]
```

The model is Not performing well in 5 epochs 

**Train Accuracy : ------------> 50.35%  (Very bad)**

**Validation Accuracy  -----> 49.58 (Better to flip a coin)**

**Test Accuracy :--------------> 56.34  ( Just got lucky. better luck next time)**



A function to Check random sentiments: 

```python
mport spacy
sp = spacy.load('en_core_web_sm')

def predict(sentence):
    if type(sentence) == str:
        tokanized_sentence = [word.text for word in sp.tokenizer(sentence)]
    else:
        tokanized_sentence = sentence


    input_data = [TEXT.vocab.stoi[word.lower()] for word in tokanized_sentence]
    input_data = torch.tensor(input_data, dtype=torch.int64).unsqueeze(1).to(device)


    model.eval()
    output = model(input_data)
    # print(output)
    predict = output.argmax(1)
    predict = predict.squeeze(0)
    print(output)

    if predict>0:
        return "---->> Positive Review"
    else:
        return '---->> Negative Review'
```

```python
In [31]:
predict('Very bad') # predict funciton will predict if this is positive or negative review.
tensor([[ 0.9696, -0.5756]], device='cuda:0', grad_fn=<AddmmBackward>)
Out[31]:
'---->> Negative Review'
In [32]:
predict('Very good') # predict funciton will predict if this is positive or negative review.
tensor([[0.0022, 0.0456]], device='cuda:0', grad_fn=<AddmmBackward>)
Out[32]:
'---->> Positive Review'
In [34]:
predict('i recommend to watch the movie once. It is mindblowing') # predict funciton will predict if this is positive or negative review.
tensor([[ 0.2117, -0.0614]], device='cuda:0', grad_fn=<AddmmBackward>)
Out[34]:
'---->> Negative Review'
```



##### **<u>Fun Fact:</u>**

 I have a NVIDIA GTX 960m with 4GB RAM.  It took lot of time to run in my local machine  and it crashed min 10 times  with batch size 16,32, 64 . So i had to run it in colab and wait till the **Spacy tokenizer** to finish the task (After lot of googling i figured it out ). Hence we modified it to custom tokenizer as my intention to build the skeleton first. 

Still  **IT'S JUST NOT PERFORMING WELL**. Is it too soon to tell that ? , Do we need to run it for more epochs ? What to do to improve the model ?

Keep experimenting  :-)

