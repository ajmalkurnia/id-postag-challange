from model.crf import CRFTagger
from model.dl_hybrid_tagger import DLHybridTagger
from common import datautils
from argparse import ArgumentParser
from sklearn.metrics import classification_report
import logging
import os


def process_crf(data):
    train, test, valid = data
    train += valid
    if args.loadmodel:
        logging.info("Load model")
        crf_tagger = CRFTagger.load(args.loadmodel)
    else:
        logging.info("CRF Training")
        crf_tagger = CRFTagger()
        crf_tagger.train(train[0], train[1])
    logging.info("Prediction")
    y_pred = crf_tagger.predict(test[0])
    if args.savemodel:
        logging.info("Save model")
        crf_tagger.save(args.savemodel)
    return y_pred


def process_cnn_rnn_crf(data):
    train, test, valid = data
    if args.loadmodel:
        logging.info("Load model")
        hybrid_tagger = DLHybridTagger.load(args.loadmodel)
    else:
        logging.info("CNN-RNN-CRF Training")
        seq_length = {
            "idn_tagged_corpus": 90,
            "ud_id": 190
        }
        hybrid_tagger = DLHybridTagger(
            word_embed_file=args.embeddingfile, we_type=args.embeddingtype,
            seq_length=seq_length[args.dataset], crf=True, char_embedding="cnn"
        )
        hybrid_tagger.train(
            train[0], train[1], args.epoch, valid
        )
    logging.info("Prediction")
    y_pred = hybrid_tagger.predict(test[0])
    if args.savemodel:
        logging.info("Save model")
        hybrid_tagger.save(args.savemodel)
    return y_pred


def evaluate(pred, ref):
    labels = set([tag for row in ref for tag in row])
    predictions = [tag for row in pred for tag in row]
    truths = [tag for row in ref for tag in row]
    report = classification_report(
        truths, predictions,
        target_names=sorted(list(labels)), digits=4
    )
    return report


def init_log(args):
    log_parameter = {
        "level": logging.INFO,
    }
    if args.logfile:
        log_parameter["filename"] = args.logfile
        log_parameter["filemode"] = "w"

    logging.basicConfig(**log_parameter)
    if args.logfile:
        consolelog = logging.StreamHandler()
        consolelog.setLevel(logging.DEBUG)
        logging.getLogger("").addHandler(consolelog)


def main(args):
    init_log(args)
    if not os.path.isdir("data/"):
        os.makedirs("data/")

    logging.info("Fetching dataset")
    data_path = datautils.DATA_CLASS[args.dataset].retrieve_data()
    data_instance = datautils.DATA_CLASS[args.dataset](data_path)

    logging.info("Parse dataset")
    dataset = data_instance.get_data()

    if args.model == "crf":
        prediction = process_crf(dataset)
    elif args.model == "cnn_rnn_crf":
        prediction = process_cnn_rnn_crf(dataset)

    report = evaluate(prediction, dataset[1][1])
    logging.info("Evaluation Results")
    logging.info(f"\n{report}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, help="Choose Dataset",
        choices={"idn_tagged_corpus", "ud_id"}, default="idn_tagged_corpus"
    )
    parser.add_argument(
        "-m", "--model", type=str, help="Choose Model",
        choices={"crf", "cnn_rnn_crf"}, default="crf"
    )
    parser.add_argument(
        "--embeddingtype", type=str,
        help="Word Embedding (Deep learning only)",
        choices={"w2v", "ft", "glove", "glorot_uniform"},
        default="glorot_uniform"
    )
    parser.add_argument(
        "--embeddingfile", type=str,
        help="Path to Word embedding pretrained model (Deep learning only)"
    )

    parser.add_argument(
        "--epoch", type=int, default=100,
        help="Training epoch (Deep learning only)"
    )
    parser.add_argument("--savemodel", type=str, help="Path to save model")
    parser.add_argument("--loadmodel", type=str, help="Path to load model")
    parser.add_argument("--logfile", type=str, help="Log file")
    args = parser.parse_args()
    main(args)
