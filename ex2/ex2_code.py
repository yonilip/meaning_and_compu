import os

import gensim
import scipy.stats
import logging
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

window_sizes = [2, 10]
dims = [100, 1000]

sentences = []
words = []


def parse_corpus():
    global sentences
    global words

    logger.info("Parsing corpus")
    if os.path.exists("sentences.txt") and os.path.exists("words.txt"):
        logger.info("Already parsed, loading pickles")
        with open("sentences.txt", 'rb') as fp:
            sentences = pickle.load(fp)
        with open("words.txt", 'rb') as fp:
            words = pickle.load(fp)
        return

    with open("corpus_ex2") as f:
        for word in f:
            word = word.strip()

            if word == '</s>':
                sentences.append(words)
                words = []
            elif word == '<s>' or word.startswith('<text'):
                continue
            else:
                words.append(word)
    logger.info("Finished parsing, pickling")
    with open("sentences.txt", 'wb') as fp:
        pickle.dump(sentences, fp)
    with open("words.txt", 'wb') as fp:
        pickle.dump(words, fp)
    logger.info("Finished pickling")
    return


def train_model(window, size):
    model_file = "model_file_{win}_{size}".format(win=window, size=size)
    if os.path.exists(model_file):
        logger.info("{model} exists, loading".format(model=model_file))
        return gensim.models.Word2Vec.load(model_file)

    model = gensim.models.Word2Vec(sentences=sentences, min_count=5, window=window, size=size)
    model.save(model_file)
    return model


def calc_sim(model, size, window):
    pair_list = []
    with open(r"SimLex-999/SimLex-999.txt") as f:
        with open('size{size}_window{window}'.format(size=size, window=window), 'w') as out:
            for line in f:
                word1, word2, POS, SimLex999, conc_w1, conc_w2, concQ, Assoc_USF, SimAssoc333, SD_SimLex = line.split()
                if word1 == 'word1':
                    continue
                try:
                    our_sim = model.similarity(word1, word2)
                except Exception as e:
                    logger.warning("caught exception")
                    # logger.exception(e.message)
                    continue
                pair_list.append((word1, word2, POS, our_sim, SimLex999))
                out.write("{w1} {w2} {pos} {sim_score}\n".format(w1=word1, w2=word2,
                                                                 pos=POS, sim_score=our_sim))

            zipped_list = zip(*pair_list)
            our_sim_vec = list(zipped_list[3])
            simlex_vec = list(zipped_list[4])
            rho, p_val = scipy.stats.spearmanr(our_sim_vec, simlex_vec)
            out.write(
                "For all POS, Spearman Co-efficient and P-Value are: {rho}, {p_val}\n".format(rho=rho, p_val=p_val))
            for pos in ['A', 'N', 'V']:
                zipped_list_by_pos = zip(*filter(lambda tup: tup[2] == pos, pair_list))
                our_sim_vec_by_pos = list(zipped_list_by_pos[3])
                simlex_vec_by_pos = list(zipped_list_by_pos[4])
                rho, p_val = scipy.stats.spearmanr(our_sim_vec_by_pos, simlex_vec_by_pos)
                out.write("For POS {pos}, "
                          "Spearman Co-efficient and P-Value are: {rho}, {p_val}\n".format(rho=rho, p_val=p_val,
                                                                                           pos=pos))


def analogy_check(model, window, dim):
    with open("analogies.txt", 'a') as f:
        # analogies: each item is tuple of lists of pos and neg and expected
        analogies = [(['army', 'tree'], ['soldier'], ['forest']),
                     (['candy', 'salt'], ['sweet'], ['snack']),
                     (['plane', 'rail'], ['sky'], ['train']),
                     (['music', 'mouth'], ['ears'], ['food']),
                     (['conductor', 'movie'], ['orchestra'], ['director'])]
        for analogy in analogies:
            res = model.most_similar(positive=analogy[0], negative=analogy[1])
            f.write("For window {window} and dim {dim}, Positive: {pos}, Negative: {neg}\n"
                    "We expected result to be: {result}\n"
                    "Top 3 Results:\n"
                    "{res_1}\n"
                    "{res_2}\n"
                    "{res_3}\n\n"
                    .format(window=window, dim=dim, pos=analogy[0], neg=analogy[1], result=analogy[2], res_1=res[0],
                            res_2=res[1], res_3=res[2]))


if __name__ == '__main__':
    parse_corpus()
    for window in window_sizes:
        for dim in dims:
            model = train_model(window, dim)
            calc_sim(model, dim, window)
            analogy_check(model, window, dim)
