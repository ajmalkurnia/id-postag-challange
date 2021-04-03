# ID POS Tagging

Explore pos-tagging on various indonesian dataset using crf and deep learing method.

## Dataset
 
1. IDN Tagger Corpus (IDN) [Link](https://github.com/famrashel/idn-tagged-corpus)
2. Indonesian Dependency Corpus (UD ID) [Link](https://github.com/UniversalDependencies/UD_Indonesian-GSD)

## Model

1. CRF
2. CNN-BiLSTM-CRF

### 1. CRF

Implementation  : sklearn_crfsuite
Hyperparameter  : default
Features        : 
 - Character [1, 6]-gram with added `«` and `»` in the beginning and end of a word
 - Relative position of a token, calculated with `pos/len(text)` 
 - 3-token surrounding current character with `["BOS"]` and `["EOS"]` added in the beginning and ending of a sentence
 - Various morphology features, such as, `is_alpha`, `is_numeric`, `is_punct`, `has_alpha`, `has_numeric`, `has_punct`, `is_upper_case`, `is_init_upper_case`

#### 2. CNN-BiLSTM-CRF
Implementation  : Keras + Tensorflow Addon (for CRF layer)
Hyperparameter  : 
- CNN embedding config : 3 seperate conv. layer 
  - #filter = 30, filter_size = 2
  - #filter = 30, filter_size = 3
  - #filter = 30, filter_size = 4 
- Word embedding config : pretrained [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) from official fasttext
- LSTM Unit : 100
- dropout : 0.5 (after char embedding, after RNN)
- optimizer : "adam"
- Vocab size : 10k
Architecture    :
  - char_embedding : [Char level input] -> CNN -> Concat all CNN -> GlobalMaxPooling1D -> Dropout
  - Sequence : [Word level input] -> Pretrained Word embedding -> Dropout -> Concat(char_embedding, word_embedding) -> BiLSTM -> Dropout -> CRF

## Results

Due to data imbalances, all listed performance is calculated with weighted-macro-average instead of reguler macro-average

### IDN Tagger Corpus

| Method      | Precision | Recall | F1-score |
| ----------- | --------- | ------ | -------- |
| CRF         | 0.9723    | 0.9724 | 0.9721   |
| CNN-RNN-CRF | 0.9672    | 0.9670 | 0.9667   |

### UD ID

| Method      | Precision | Recall | F1-score |
| ----------- | --------- | ------ | -------- |
| CRF         | 0.9368    | 0.9367 | 0.9366   |
| CNN-RNN-CRF | 0.9263    | 0.9260 | 0.9256   |

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
