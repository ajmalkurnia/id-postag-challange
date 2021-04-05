# POS-Tag Indonesia

Eksplorasi *task pos-tagging* untuk dataset bahasa Indonesia dengan menggunakan CRF dan metode *deep learning* lainnya.

## Dataset
 
1. IDN Tagger Corpus (IDN) [Link](https://github.com/famrashel/idn-tagged-corpus)
2. Indonesian Dependency Corpus (UD ID) [Link](https://github.com/UniversalDependencies/UD_Indonesian-GSD)

| Stat.           | IDN     | UD ID  |
| --------------- | ------- | ------ |
| #Kalimat        | 10030   | 5593   |
| #Kata           | 256622  | 121923 |
| #Karakter       | 1427666 | 629578 |
| #Kata unik      | 18287   | 22221  |
| # Label POS-Tag | 24      | 16     |

## Model

1. CRF
2. RNN
3. CNN-RNN
4. RNN-CRF
5. CNN-RNN-CRF

Hasil pelatihan kelima model dapat diunduh di [sini](https://drive.google.com/drive/folders/1BeZm01S7K7Uo-La6pVvT_mETefF6Y3Hh?usp=sharing)

### 1. CRF

Implementasi  : sklearn_crfsuite

Hyperparameter  : default

Fitur        : 
 - Karakter [1, 6]-gram dengan tambahan karakter `«` and `»` diawal dan diakhir sebuah kata.
 - Posisi relatif token pada kalimat, dihitung dengan  `pos/len(text)` 
 - 3-token di sekeliling token saat ini dengan tamahan token khusus `["BOS"]` dan `["EOS"]` di awal dan di akhir kalimat  
 - Berbagai macam fitur morfologi, seperti, `is_alpha`, `is_numeric`, `is_punct`, `has_alpha`, `has_numeric`, `has_punct`, `is_upper_case`, `is_init_upper_case`

#### 2. RNN
Implementasi  : Keras

Hyperparameter  : 
- *Embedding* kata : *pretrained* [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) dari official fasttext
- RNN : LSTM dengan 100 unit, dua arah (*bidirectional*)
- Dropout : 0.5 (setelah *Embedding*, didalam RNN, setelah RNN)
- Optimizer : "adam"
- Ukuran kamus : 10k
- Train Epoch : 100 + *early stopping* dengan 10 *patience* berdasarkan *loss* data validasi

Arsitektur    :
  - *Sequence* : [Input kata] -> Pretrained *Embedding* kata -> Dropout -> BiLSTM -> Dropout -> Softmax

#### 3. CNN-RNN
Implementasi  : Keras

Hyperparameter  : 
- Pengaturan CNN : 3  *conv. layer* terpisah 
  - Jumlah filter = 30, Ukuran filter = 2
  - Jumlah filter = 30, Ukuran filter = 3
  - Jumlah filter = 30, Ukuran filter = 4 
- *Embedding* kata : *pretrained* [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) dari official fasttext
- RNN : LSTM dengan 100 unit, dua arah (*bidirectional*)
- Dropout : 0.5 (setelah *Embedding*, didalam RNN, setelah RNN)
- Optimizer : "adam"
- Ukuran kamus : 10k
- Train Epoch : 100 + *early stopping* dengan 10 *patience* berdasarkan *loss* data validasi

Arsitektur    :
  - *Embedding* karakter : [Input karakter] -> CNN -> Concat all CNN -> GlobalMaxPooling1D -> Dropout
  - *Sequence* : [Input kata] -> Pretrained *Embedding* kata -> Dropout -> Concat(*embedding* karakter, *embedding* kata) -> BiLSTM -> Dropout -> Softmax

#### 4. RNN-CRF
Implementasi  : Keras + Tensorflow Addon (CRF layer)

Hyperparameter  : 
- *Embedding* kata : *pretrained* [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) dari official fasttext
- RNN : LSTM dengan 100 unit, dua arah (*bidirectional*)
- dropout : 0.5 (setelah *Embedding*, didalam RNN, setelah RNN)
- optimizer : "adam"
- ukuran kamus : 10k
- Train Epoch : 100 + *early stopping* dengan 10 *patience* berdasarkan *loss* data validasi

Arsitektur    :
  - *embedding* karakter : [Input karakter] -> CNN -> Concat semua CNN -> GlobalMaxPooling1D -> Dropout
  - *Sequence* : [Input kata] -> Pretrained *Embedding* kata -> Dropout -> Concat(embedding karakter, *embedding* kata) -> BiLSTM -> Dropout -> CRF

#### 5. CNN-RNN-CRF
Implementasi  : Keras + Tensorflow Addon (CRF layer)

Hyperparameter  : 
- CNN *embedding* config : 3 seperate conv. layer 
  - Jumlah filter = 30, Ukuran filter = 2
  - Jumlah filter = 30, Ukuran filter = 3
  - Jumlah filter = 30, Ukuran filter = 4 
- *Embedding* kata : *pretrained* [wiki.id](https://fasttext.cc/docs/en/pretrained-vectors.html) dari official fasttext
- RNN : LSTM dengan 100 unit, dua arah (*bidirectional*)
- dropout : 0.5 (setelah *Embedding*, didalam RNN, setelah RNN
- optimizer : "adam"
- ukuran kamus : 10k
- Train Epoch : 100 + *early stopping* dengan 10 *patience* berdasarkan *loss* data validasi
  
Arsitektur    :
  - *embedding* karakter : [Input karakter] -> CNN -> Concat all CNN -> GlobalMaxPooling1D -> Dropout
  - *Sequence* : [Input kata] -> Pretrained *Embedding* kata -> Dropout -> Concat(*embedding* karakter, *embedding* kata) -> BiLSTM -> Dropout -> CRF

## Demo

- Contoh menggunakan model crf untuk dataset idn
```
python3 -d idn_tagged_dataset -m crf
```
- Simpan model
```
python3 -d idn_tagged_dataset -m crf --savemodel PATH/TO/MODEL.zip
```
- Load model
```
python3 -d idn_tagged_dataset -m crf --loadmodel PATH/TO/MODEL.zip
```
- Pengaturan lain
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

## Hasil

Karena jumlah label yang tidak seimbang, semua hasil yang tertulis dihitung dengan menggunakan **weighted-macro-average**

### IDN Tagger Corpus

Dataset dipisah 80:20 untuk pelatihan dan testing. Data validasi hanya digunakan pada metode berbasiskan deep learning dan diambil dari 10% data latih 

| Method      | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- |
| CRF         | **0.9723** | **0.9724** | **0.9721** |
| RNN         | 0.9640     | 0.9639     | 0.9637     |
| CNN-RNN     | 0.9693     | 0.9694     | 0.9692     |
| RNN-CRF     | 0.9645     | 0.9644     | 0.9641     |
| CNN-RNN-CRF | 0.9694     | 0.9694     | 0.9691     |

### UD ID

Pembagian pada dataset ini mengikuti pembagian dari dataset aslinya. Untuk model CRF data dev/validasi digunakan sebagai tambahan data latih

| Method      | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- |
| CRF         | **0.9368** | **0.9367** | **0.9366** |
| RNN         | 0.9118     | 0.9083     | 0.9083     |
| CNN-RNN     | 0.9278     | 0.9275     | 0.9271     |
| RNN-CRF     | 0.9109     | 0.9074     | 0.9074     |
| CNN-RNN-CRF | 0.9290     | 0.9287     | 0.9284     |

## Referensi
1. [CNN-RNN-CRF](https://www.aclweb.org/anthology/P16-1101/)


## *Package/library*
- tensorflow==2.4.1
- tensorflow-addons==0.12.1
- sklearn-crfsuite==0.3.6
- scikit-learn==0.20.3
- numpy==1.19.5
- gensim==4.0.0
- fasttext==0.9.2
