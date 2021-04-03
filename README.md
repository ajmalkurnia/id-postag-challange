# ID POS Tagging

Explore pos-tagging on various indonesian dataset using crf and deep learing method.

## Dataset
 
1. IDN Tagger Corpus (IDN) [Link](https://github.com/famrashel/idn-tagged-corpus)
2. Indonesian Dependency Corpus (UD ID) [Link](https://github.com/UniversalDependencies/UD_Indonesian-GSD)

| Stat.          | IDN     | UD ID  |
| -------------- | ------- | ------ |
| #Sentences     | 10030   | 5593   |
| #Words         | 256622  | 121923 |
| #Characters    | 1427666 | 629578 |
| Unique words   | 18287   | 22221  |
| Unique POS-Tag | 24      | 16     |

## Model

1. CRF
2. RNN
3. CNN-RNN
4. RNN-CRF
5. CNN-RNN-CRF

The link of pretrained model will be provided

### 1. CRF

Implementation  : sklearn_crfsuite

Hyperparameter  : default

Features        : 
 - Character [1, 6]-gram with added `«` and `»` in the beginning and end of a word
 - Relative position of a token, calculated with `pos/len(text)` 
 - 3-token surrounding current character with `["BOS"]` and `["EOS"]` added in the beginning and ending of a sentence
 - Various morphology features, such as, `is_alpha`, `is_numeric`, `is_punct`, `has_alpha`, `has_numeric`, `has_punct`, `is_upper_case`, `is_init_upper_case`

#### 2. RNN
Implementation  : Keras

Hyperparameter  : 
- Word embedding config : pretrained [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) from official fasttext
- RNN : LSTM with 100 unit, bidirectional
- dropout : 0.5 (after Embedding, inside RNN, after RNN)
- optimizer : "adam"
- Vocab size : 10k
- Train Epoch : 100 + early stopping with 10 patience based on validation

Architecture    :
  - Sequence : [Word level input] -> Pretrained Word embedding -> Dropout -> BiLSTM -> Dropout -> Softmax

#### 3. CNN-RNN
Implementation  : Keras + Tensorflow Addon (for CRF layer)

Hyperparameter  : 
- CNN embedding config : 3 seperate conv. layer 
  - #filter = 30, filter_size = 2
  - #filter = 30, filter_size = 3
  - #filter = 30, filter_size = 4 
- Word embedding config : pretrained [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) from official fasttext
- RNN : LSTM with 100 unit, bidirectional
- dropout : 0.5 (after Embedding, inside RNN, after RNN)
- optimizer : "adam"
- Vocab size : 10k
- Train Epoch : 100 + early stopping with 10 patience based on validation

Architecture    :
  - char_embedding : [Char level input] -> CNN -> Concat all CNN -> GlobalMaxPooling1D -> Dropout
  - Sequence : [Word level input] -> Pretrained Word embedding -> Dropout -> Concat(char_embedding, word_embedding) -> BiLSTM -> Dropout -> Softmax

#### 4. RNN-CRF
Implementation  : Keras + Tensorflow Addon (for CRF layer)

Hyperparameter  : 
- Word embedding config : pretrained [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) from official fasttext
- RNN : LSTM with 100 unit, bidirectional
- dropout : 0.5 (after Embedding, inside RNN, after RNN)
- optimizer : "adam"
- Vocab size : 10k
- Train Epoch : 100 + early stopping with 10 patience based on validation

Architecture    :
  - char_embedding : [Char level input] -> CNN -> Concat all CNN -> GlobalMaxPooling1D -> Dropout
  - Sequence : [Word level input] -> Pretrained Word embedding -> Dropout -> Concat(char_embedding, word_embedding) -> BiLSTM -> Dropout -> CRF

#### 5. CNN-RNN-CRF
Implementation  : Keras + Tensorflow Addon (for CRF layer)

Hyperparameter  : 
- CNN embedding config : 3 seperate conv. layer 
  - #filter = 30, filter_size = 2
  - #filter = 30, filter_size = 3
  - #filter = 30, filter_size = 4 
- Word embedding config : pretrained [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) from official fasttext
- RNN : LSTM with 100 unit, bidirectional
- dropout : 0.5 (after Embedding, inside RNN, after RNN
- optimizer : "adam"
- Vocab size : 10k
- Train Epoch : 100 + early stopping with 10 patience based on validation
  
Architecture    :
  - char_embedding : [Char level input] -> CNN -> Concat all CNN -> GlobalMaxPooling1D -> Dropout
  - Sequence : [Word level input] -> Pretrained Word embedding -> Dropout -> Concat(char_embedding, word_embedding) -> BiLSTM -> Dropout -> CRF

## Demo

- Run the crf training on idn_tagged_dataset
```
python3 -d idn_tagged_dataset -m crf
```
- Saving model
```
python3 -d idn_tagged_dataset -m crf --savemodel PATH/TO/MODEL.zip
```
- Load model
```
python3 -d idn_tagged_dataset -m crf --loadmodel PATH/TO/MODEL.zip
```
- Other configurations
```
main.py [-h] [--dataset {idn_tagged_corpus, ud_id}]
               [--model {crf, cnn_rnn, cnn_rnn_crf, rnn, rnn_crf}]
               [--embeddingtype {w2v, glove, ft, glorot_uniform}]
               [--embeddingfile PATH/TO/PRETRAINED/WORD/VECTOR.bin]
               [--epoch TRAINING_EPOCH]
               [--savemodel PATH/TO/MODEL.zip]
               [--loadmodel PATH/TO/MODEL.zip]
               [--logfile LOGFILE]
```

## Results

Due to data imbalances, all listed performance is calculated with **weighted-macro-average** instead of reguler macro-average

### IDN Tagger Corpus

| Method      | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- |
| CRF         | **0.9723** | **0.9724** | **0.9721** |
| RNN         | 0.9640     | 0.9639     | 0.9637     |
| CNN-RNN     | 0.9693     | 0.9694     | 0.9692     |
| RNN-CRF     | 0.9645     | 0.9644     | 0.9641     |
| CNN-RNN-CRF | 0.9694     | 0.9694     | 0.9691     |

### UD ID

| Method      | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- |
| CRF         | **0.9368** | **0.9367** | **0.9366** |
| RNN         | 0.9118     | 0.9083     | 0.9083     |
| CNN-RNN     | 0.9278     | 0.9275     | 0.9271     |
| RNN-CRF     | 0.9109     | 0.9074     | 0.9074     |
| CNN-RNN-CRF | 0.9290     | 0.9287     | 0.9284     |

## References
1. [CNN-RNN-CRF](https://www.aclweb.org/anthology/P16-1101/)


## Requirement
- tensorflow==2.4.1
- tensorflow-addons==0.12.1
- sklearn-crfsuite==0.3.6
- scikit-learn==0.20.3
- numpy==1.19.5
- gensim==4.0.0
- fasttext==0.9.2
