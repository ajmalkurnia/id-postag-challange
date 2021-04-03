from keras.layers import LSTM, Embedding, TimeDistributed, Concatenate
from keras.layers import Dropout, Bidirectional, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPool1D, Dense
from keras.utils import to_categorical
from keras.initializers import Constant, RandomUniform
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input, load_model
from keras.callbacks import EarlyStopping
from tensorflow_addons.layers.crf import CRF

from common.word_vector import WE_TYPE
from .tf_model_crf import ModelWithCRFLoss

from tempfile import TemporaryDirectory
from collections import defaultdict
from zipfile import ZipFile
import numpy as np
import pickle
import string
import os


class CNNRNNCRFTagger():
    def __init__(
        self, seq_length=100, word_length=50, char_embed_size=30,
        word_embed_size=100, word_embed_file=None, we_type="glorot_normal",
        recurrent_dropout=0.5, embedding_dropout=0.5, rnn_units=100,
        optimizer="adam", loss="categorical_crossentropy", vocab_size=10000,
        pre_crf_dropout=0.5, char_embedding="cnn", crf=True,
        conv_layers=[[30, 3, -1], [30, 2, -1], [30, 4, -1]],
    ):

        self.seq_length = seq_length
        self.word_length = word_length

        self.char_embed_size = char_embed_size
        # Will be overide with pretrained file embedding
        self.word_embed_size = word_embed_size
        self.word_embed_file = word_embed_file
        self.we_type = we_type

        self.rnn_units = rnn_units
        self.rd = recurrent_dropout
        self.ed = embedding_dropout

        self.pre_crf_dropout = pre_crf_dropout
        self.crf = crf

        self.char_embedding = char_embedding
        self.conv_layers = conv_layers

        self.loss = loss
        self.optimizer = optimizer
        self.vocab_size = vocab_size

    def __get_char_embedding(self):
        word_input_layer = Input(shape=(self.word_length, ))
        # +1 for padding
        embedding_block = Embedding(
            self.n_chars+1, self.char_embed_size,
            input_length=self.word_length, trainable=True,
            embeddings_initializer=RandomUniform(
                minval=-1*np.sqrt(3/self.char_embed_size),
                maxval=np.sqrt(3/self.char_embed_size)
            )
        )(word_input_layer)
        conv_layers = []
        for filter_num, filter_size, pooling_size in self.conv_layers:
            conv_layer = Conv1D(
                filter_num, filter_size, activation="relu"
            )(embedding_block)
            if pooling_size != -1:
                conv_layer = MaxPooling1D(
                    pool_size=pooling_size
                )(conv_layer)
            conv_layers.append(conv_layer)
        embedding_block = Concatenate(axis=1)(conv_layers)
        embedding_block = GlobalMaxPool1D()(embedding_block)
        embedding_block = Dropout(self.ed)(embedding_block)
        embedding_block = Model(
            inputs=word_input_layer, outputs=embedding_block)
        embedding_block.summary()
        seq_inp_layer = Input(
            shape=(self.seq_length, self.word_length), name="char"
        )
        embedding_block = TimeDistributed(embedding_block)(seq_inp_layer)
        return seq_inp_layer, embedding_block

    def __init_model(self):
        input_word_layer = Input(shape=(self.seq_length,), name="word")
        word_embed_block = Embedding(
            self.vocab_size+1, self.word_embed_size,
            input_length=self.seq_length,
            embeddings_initializer=self.word_embedding,
        )
        word_embed_block = word_embed_block(input_word_layer)
        word_embed_block = Dropout(self.ed)(word_embed_block)
        if self.char_embedding == "cnn":
            input_char_layer, char_embed_block = self.__get_char_embedding()
            input_layer = [input_char_layer, input_word_layer]
            embed_block = Concatenate()([char_embed_block, word_embed_block])
        else:
            embed_block = word_embed_block
            input_layer = input_word_layer
        self.model = Bidirectional(LSTM(
            units=self.rnn_units, return_sequences=True,
            dropout=self.rd,
        ))(embed_block)
        self.model = Dropout(self.pre_crf_dropout)(self.model)
        if self.crf:
            crf = CRF(self.n_label+1)
            out = crf(self.model)
            self.model = Model(
                inputs=input_layer, outputs=out
            )
            self.model.summary()
            self.model = ModelWithCRFLoss(self.model)
        else:
            out = TimeDistributed(Dense(
                self.n_label+1, activation="softmax"
            ))(self.model)
            self.model = Model(input_layer, out)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer
        )

    def __init_c2i(self):
        vocab = set([*string.printable])
        self.n_chars = len(vocab)
        self.char2idx = {ch: i+1 for i, ch in enumerate(vocab)}

    def __init_w2i(self, data):
        vocab = defaultdict(int)
        for s in data:
            for w in s:
                vocab[w] += 1
        vocab = sorted(vocab.items(), key=lambda d: (d[1], d[0]))
        vocab = [v[0] for v in vocab]
        vocab = list(reversed(vocab))[:self.vocab_size]
        self.n_words = len(vocab)
        self.word2idx = {word: idx+1 for idx, word in enumerate(vocab)}

    def __init_l2i(self, data):
        label = list(set([lb for sub in data for lb in sub]))
        self.n_label = len(label)
        self.label2idx = {ch: idx+1 for idx, ch in enumerate(sorted(label))}
        self.idx2label = {idx: ch for ch, idx in self.label2idx.items()}

    def __init_wv_embedding(self):
        """
        Initialization of for Word embedding matrix
        UNK word will be initialized randomly
        """
        wv_model = WE_TYPE[self.we_type].load_model(self.word_embed_file)
        self.word_embed_size = wv_model.size

        self.word_embedding = np.zeros(
            (self.vocab_size+1, wv_model.size), dtype=float
        )
        for word, idx in self.word2idx.items():
            self.word_embedding[idx, :] = wv_model.retrieve_vector(word)

    def __init_embedding(self):
        if self.we_type in ["w2v", "ft", "glove"]:
            self.__init_wv_embedding()
            self.word_embedding = Constant(self.word_embedding)
        else:
            self.word_embedding = self.we_type

    def __char_vector(self, inp_seq):
        vector_seq = np.zeros(
            (len(inp_seq), self.seq_length, self.word_length)
        )
        for i, data in enumerate(inp_seq):
            data = data[:self.seq_length]
            for j, word in enumerate(data):
                word = word[:self.word_length]
                for k, ch in enumerate(word):
                    if ch in self.char2idx:
                        vector_seq[i, j, k] = self.char2idx[ch]
        return vector_seq

    def __word_vector(self, inp_seq):
        vector_seq = np.zeros((len(inp_seq), self.seq_length))
        for i, data in enumerate(inp_seq):
            data = data[:self.seq_length]
            for j, word in enumerate(data):
                if word in self.word2idx:
                    vector_seq[i][j] = self.word2idx[word]
        return vector_seq

    def vectorize_input(self, inp_seq):
        word_vector = self.__word_vector(inp_seq)
        if self.char_embedding:
            char_vector = self.__char_vector(inp_seq)
        else:
            char_vector = None
        return word_vector, char_vector

    def vectorize_label(self, out_seq):
        out_seq = [[self.label2idx[w] for w in s] for s in out_seq]
        out_seq = pad_sequences(
            maxlen=self.seq_length, sequences=out_seq, padding="post"
        )
        if not self.crf:
            out_seq = [
                to_categorical(i, num_classes=self.n_label) for i in out_seq
            ]
        #  To Category
        return np.array(out_seq)

    def get_crf_label(self, pred_sequence, input_data):
        label_seq = []
        for i, s in enumerate(pred_sequence[0]):
            tmp = []
            for j, w in enumerate(s[:len(input_data[i])]):
                if w in self.idx2label:
                    label = self.idx2label[w]
                else:
                    label = self.idx2label[np.argmax(
                        pred_sequence[1][i][j][1:]
                    ) + 1]
                tmp.append(label)
            label_seq.append(tmp)
        return label_seq

    def get_greedy_label(self, pred_sequence, input_data):
        label_seq = []
        for i, s in enumerate(pred_sequence):
            tmp_pred = []
            for j, w in enumerate(s):
                if j < len(input_data[i]):
                    tmp_pred.append(self.idx2label[np.argmax(w[1:])+1])
            label_seq.append(tmp_pred)
        return label_seq

    def devectorize_label(self, pred_sequence, input_data):
        if self.crf:
            return self.get_crf_label(pred_sequence, input_data)
        else:
            return self.get_greedy_label(pred_sequence, input_data)

    def prepare_data(self, X, y=None):
        vector_word_X, vector_char_X = self.vectorize_input(X)
        X_input = {"word": vector_word_X}
        if self.char_embedding:
            X_input["char"] = vector_char_X

        vector_y = None
        if y is not None:
            vector_y = self.vectorize_label(y)
        return X_input, vector_y

    def train(self, X, y, n_epoch, valid_split, batch_size=32):
        self.__init_l2i(y)
        if self.char_embedding:
            self.__init_c2i()

        self.__init_w2i(X)
        self.__init_embedding()
        self.__init_model()

        X_train, y_train = self.prepare_data(X, y)
        X_valid, y_valid = self.prepare_data(valid_split[0], valid_split[1])
        es = EarlyStopping(
            monitor="val_crf_loss" if self.crf else "val_loss",
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True
        )
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=n_epoch,
            validation_data=(X_valid, y_valid),
            verbose=1,
            callbacks=[es]
        )
        return history

    def predict(self, X):
        X_test, _ = self.prepare_data(X)
        pred_result = self.model.predict(X_test)
        label_result = self.devectorize_label(pred_result, X)
        return label_result

    def save(self, filepath):
        """
        Write the model and class parameter into a zip file
        :param filepath: string, path of saved file with ".zip" format
        """
        filename = filepath.split("/")[-1].split(".")[0]
        filenames = {
            "model": f"{filename}_network",
            "class_param": f"{filename}_class.pkl"
        }
        with TemporaryDirectory() as tmp_dir:
            network_path = f"{tmp_dir}/{filenames['model']}"
            self.model.save(network_path, save_format="tf")
            class_param = {
                "label2idx": self.label2idx,
                "word2idx": self.word2idx,
                "char2idx": self.char2idx,
                "seq_length": self.seq_length,
                "word_length": self.word_length,
                "idx2label": self.idx2label
            }
            with open(f"{tmp_dir}/{filenames['class_param']}", "wb") as pkl:
                pickle.dump(class_param, pkl)
            with ZipFile(filepath, "w") as zipf:
                zipf.write(
                    f"{tmp_dir}/{filenames['class_param']}",
                    filenames['class_param']
                )
                for dirpath, dirs, files in os.walk(network_path):
                    if files == []:
                        zipf.write(
                            dirpath, "/".join(dirpath.split("/")[-2:])+"/"
                        )
                    for f in files:
                        fn = os.path.join(dirpath, f)
                        zipf.write(fn, "/".join(fn.split("/")[3:]))

    @staticmethod
    def load(filepath):
        """
        Load model from the saved zipfile
        :param filepath: path to model zip file
        """
        with ZipFile(filepath, "r") as zipf:
            filelist = zipf.filelist
            model_dir = ""
            with TemporaryDirectory() as tmp_dir:
                for fn in filelist:
                    filename = fn.filename
                    if filename.endswith("_class.pkl"):
                        with zipf.open(filename, "r") as pkl:
                            pickle_content = pkl.read()
                            class_param = pickle.loads(pickle_content)
                    elif filename.split("/")[0].endswith("_network"):
                        model_dir = filename.split("/")[0]
                        zipf.extract(filename, tmp_dir)
                model = load_model(
                    f"{tmp_dir}/{model_dir}",
                    custom_objects={
                        "ModelWithCRFLoss": ModelWithCRFLoss
                    }
                )
                model.summary()

            constructor_param = {
                "seq_length": class_param["seq_length"],
                "word_length": class_param["word_length"]
            }
            classifier = CNNRNNCRFTagger(**constructor_param)
            classifier.model = model
            classifier.label2idx = class_param["label2idx"]
            classifier.word2idx = class_param["word2idx"]
            classifier.char2idx = class_param["char2idx"]
            classifier.idx2label = class_param["idx2label"]
            classifier.n_label = len(classifier.label2idx)
        return classifier
