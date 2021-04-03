import os
import urllib
from sklearn.model_selection import train_test_split

DATA_URL = {
    "idn_tagged_corpus": "https://raw.githubusercontent.com/famrashel/idn-tagged-corpus/master/Indonesian_Manually_Tagged_Corpus_ID.tsv",  # noqa
    "ud_id": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-GSD/master/id_gsd-ud-{}.conllu"  # noqa
}


class IDNTaggedCorpus():
    def __init__(self, path):
        self.datapath = path

    def parse_data(self, path):
        data = []
        with open(path, "r") as f:
            sentences = f.read().split("\n</kalimat>")
            for sentence in sentences:
                token_sequence = []
                tag_sequenece = []
                tokens = sentence.split("\n")
                for idx, token in enumerate(tokens):
                    if len(token.split("\t")) == 2:
                        word, tag = token.split("\t")
                        token_sequence.append(word)
                        tag_sequenece.append(tag)
                if token_sequence != []:
                    data.append({
                        "tokenized_text": token_sequence,
                        "tag": tag_sequenece
                    })
        return data

    def get_data(
        self, train_split=0.8, test_split=0.2, train_valid_split=0.1,
        seed=31256
    ):
        id_treebank_dataset = self.parse_data(self.datapath)
        X = [row["tokenized_text"] for row in id_treebank_dataset]
        y = [row["tag"] for row in id_treebank_dataset]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_split, test_size=test_split,
            random_state=seed
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=1-train_valid_split,
            test_size=train_valid_split, random_state=seed
        )
        return (X_train, y_train), (X_test, y_test), (X_val, y_val)

    @staticmethod
    def retrieve_data():
        save_path = "data/Indonesian_Manually_Tagged_Corpus_ID.tsv"
        if not os.path.isfile(save_path):
            fname, headers = urllib.request.urlretrieve(
                DATA_URL["treebank"], save_path
            )
        else:
            print("file already exists")
        return save_path


class UDIDDataset():
    def __init__(self, path):
        self.datapath = path
        self.template_fname = "id_gsd-ud-{}.conllu"

    def parse_data(self, path):
        data = []
        token_sequence = []
        tag_sequence = []
        with open(path, "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                elif line.strip() != "":
                    words_detail = line.strip().split("\t")
                    token_sequence.append(words_detail[1])
                    tag_sequence.append(words_detail[3])
                else:
                    data.append({
                        "tokenized_text": token_sequence,
                        "tag": tag_sequence
                    })
                    token_sequence = []
                    tag_sequence = []
        return data

    def get_data(
        self, train_split=0, test_split=0, train_valid_split=0, seed=31256
    ):
        data = []
        for split in ["train", "test", "dev"]:
            split_data = self.parse_data(
                self.datapath + self.template_fname.format(split)
            )
            data.append((
                [row["tokenized_text"] for row in split_data],
                [row["tag"] for row in split_data]
            ))

        return tuple(data)

    @staticmethod
    def retrieve_data():
        save_dir = "data/ud_id/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        template_fname = "id_gsd-ud-{}.conllu"
        for split in ["train", "dev", "test"]:
            save_path = save_dir + template_fname.format(split)
            if not os.path.isfile(save_path):
                fname, headers = urllib.request.urlretrieve(
                    DATA_URL["ud_id"].format(split), save_path
                )
        return save_dir


DATA_CLASS = {
    "idn_tagged_corpus": IDNTaggedCorpus,
    "ud_id": UDIDDataset
}
