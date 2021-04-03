import regex as re
import string
import sklearn_crfsuite
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import pickle


class CRFTagger():
    def __init__(
        self, ngram=(2, 6), window=3, full_morph=True, position=True, **kwargs
    ):
        """
        :param window: integer, size of window feature
            the value of n-token before and n-token after
        :param ngram: tuple, inclusive range of n-gram character features
        :param full_morph: boolean, use full range of morphology features
        :param position: boolean, token use position feature
        :param kwargs: Keyword argument to be passed on sklearn_crfsuite.CRF

        """
        self.model = sklearn_crfsuite.CRF(**kwargs)
        self.ngram = ngram
        self.window = window
        self.full_morph = full_morph
        self.position = position

    def _make_n_gram(self, word):
        """
        :param word: string, a word that to be generated
        :return gram_list: dictionary, Character n-gram of the word
            for each value n within range of self.ngram
        """
        gram = {}
        word = word.strip()
        for n in range(self.ngram[0], self.ngram[1]+1):
            pattern = r'(?=(.{' + re.escape(str(n)) + r'}))'
            gram[str(n)] = list(set([*re.findall(pattern, "«"+word+"»")]))
        return gram

    def __has_alpha(self, word):
        if re.search(r"[a-zA-Z]+", word):
            return True
        else:
            return False

    def __has_digit(self, word):
        if re.search(r"[0-9]+", word):
            return True
        else:
            return False

    def __has_punct(self, word):
        if re.search(r"["+re.escape(string.punctuation)+r"]+", word):
            return True
        else:
            return False

    def __is_punct(self, word):
        if re.search(r"^["+re.escape(string.punctuation)+r"]+$", word):
            return True
        else:
            return False

    def __extract_feature(self, text):
        """
        :param text     : list of token, input text
        :return features: list of dictionary, features of text,
            the feature consist of:
                ngram: ngram char list of a token:
                is_alphabet: is the token consist of alphabet only ?
                is_digit: is the token consist of numeric only ?
                contains_alphabet: is the token have an alphabet character ?
                contains_digit: is the token have an numeric character ?
                contains_punct: is the token have an punctuation character ?
                is_capital: is the token fully capitalized ?
                is_punct: is the token consist of punctuation only ?
                is_init_caps: is the first character of the token capitalized ?
                position: position of the character in the sentences
                token[x]: context position of x from the token
        """

        features = []
        for i, token in enumerate(text):
            feature = {}
            feature["ngram"] = self._make_n_gram(token)
            if self.full_morph:
                feature["contains_punct"] = self.__has_punct(token)
                feature["is_alphabet"] = token.isalpha()
                feature["is_digit"] = token.isnumeric()
                feature["is_capital"] = token.isupper()
                feature["contains_alphabet"] = self.__has_alpha(token)
                feature["contains_digit"] = self.__has_digit(token)
                feature['is_punct'] = self.__is_punct(token)
                feature["is_init_caps"] = token[0].isupper(
                ) and not token.isupper()

            if self.position:
                feature["position"] = i/len(text)

            if self.window:
                temp_text = ["[BOS]"] + text + ["[EOS]"]
                window_b = i - self.window
                if window_b < 0:
                    window_b = 0
                window_e = i + self.window + 1
                if window_e > len(temp_text):
                    window_e = len(temp_text)
                for idx, window_tok in enumerate(temp_text[window_b:window_e]):
                    feature[f"token[{window_b+idx-i}]"] = window_tok.lower()
            else:
                feature["token0"] = token.lower()
            # print(feature)
            features.append(feature)
        return features

    def train(self, X, y):
        """
        :params X: list of token list, input data
        :params y: list of label list, label data
        """
        y_train = y
        X_train = []
        for sequence_token in X:
            X_train.append(self.__extract_feature(sequence_token))
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        :params X: list of token list, input data
        :return predict: list of label list, prediction result
        """
        X_test = []
        for sequence_token in X:
            X_test.append(self.__extract_feature(sequence_token))
        return self.model.predict(X_test)

    def save(self, filepath):
        """
        Write the model and class parameter into a zip file
        :param filepath: string, path of saved file with ".zip" format
        """
        filename = filepath.split("/")[-1].split(".")[0]
        filenames = {
            "model": f"{filename}_crf.pkl",
            "class_param": f"{filename}_class.pkl"
        }
        with TemporaryDirectory() as tmp_dir:
            crf_path = f"{tmp_dir}/{filenames['model']}"
            with open(crf_path, "wb") as pkl:
                pickle.dump(self.model, pkl)
            class_param = {
                "ngram": self.ngram,
                "window": self.window,
                "full_morph": self.full_morph,
                "position": self.position
            }
            with open(f"{tmp_dir}/{filenames['class_param']}", "wb") as pkl:
                pickle.dump(class_param, pkl)
            with ZipFile(filepath, "w") as zipf:
                for _, v in filenames.items():
                    zipf.write(f"{tmp_dir}/{v}", v)

    @staticmethod
    def load(filepath):
        """
        Load model from the saved zipfile
        :param filepath: path to model zip file
        """
        with ZipFile(filepath, "r") as zipf:
            filelist = zipf.filelist
            for fn in filelist:
                filename = fn.filename
                if filename.endswith("_crf.pkl"):
                    with zipf.open(filename, "r") as pkl:
                        pickle_content = pkl.read()
                        model = pickle.loads(pickle_content)
                elif filename.endswith("_class.pkl"):
                    with zipf.open(filename, "r") as pkl:
                        pickle_content = pkl.read()
                        class_param = pickle.loads(pickle_content)
            constructor_param = {}
            classifier = CRFTagger(**constructor_param)
            classifier.model = model
            classifier.ngram = class_param["ngram"]
            classifier.window = class_param["window"]
            classifier.full_morph = class_param["full_morph"]
            classifier.position = class_param["position"]
        return classifier
