import gensim.downloader as api
from gensim.models.word2vec import Word2Vec, KeyedVectors
import time
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

positive_lexicon_path = "./opinion-lexicon-English/positive-words.txt"
negative_lexicon_path = "./opinion-lexicon-English/negative-words.txt"

def create_lexicon_list(filepath):
    cleaned_words = []

    with open(filepath, "r") as file:
        for line in file.readlines():
            # The comments in the file start with a ;
            if line.startswith(";"):
                continue

            # They end with \n so strip any whitespace
            cleaned_word = line.strip()
            cleaned_words.append(cleaned_word)

    return cleaned_words

def prepare_vector_matrix(keyed_vectors, word_list):
    matrix = []
    valid_words = []
    parts = [[] for x in range(300)]

    for word in word_list:
        vector = None

        # Not all of the words will be in the corpus so skip them
        try:
            vector = keyed_vectors.get_vector(word)
        except KeyError:
            continue

        if vector is None:
            continue

        for i, part in enumerate(vector):
            parts[i].append(vector[i])

        matrix.append(vector)
        valid_words.append(word)

    print(len(valid_words) / len(word_list))
    # array will make them the rows so we need to transpose to get the vectors
    # as the columns
    return valid_words, np.transpose(np.array(matrix, dtype=np.float64))

def prepare_lexicon(keyed_vectors, filepath):
    words = create_lexicon_list(filepath)
    valid_words, matrix = prepare_vector_matrix(keyed_vectors, words)

    return valid_words, matrix

def prepare_keyed_vectors():
    keyed_vectors = api.load("word2vec-google-news-300")
    keyed_vectors.save("pretrained-word2vec-keyedvectors.kv")
    keyed_vectors = api.load("glove-wiki-gigaword-300")
    keyed_vectors.save("pretrained-glove-keyedvectors.kv")

def test_dsv_classifier(labelled_words, keyed_vectors, k):
    all = 0
    correct = 0

    for word in labelled_words:
        is_positive = labelled_words[word]
        projection = np.dot(k, keyed_vectors[word])
        y_star_hat = projection > 0

        if y_star_hat == is_positive:
            correct += 1

        all += 1

    return (correct, all, correct / all * 100)


word_vectors = KeyedVectors.load("pretrained-word2vec-keyedvectors.kv")
positive_words, positive_matrix = prepare_lexicon(word_vectors, positive_lexicon_path)
negative_words, negative_matrix = prepare_lexicon(word_vectors, negative_lexicon_path)

print(positive_matrix.shape, negative_matrix.shape)

pca = PCA()
pca.fit(positive_matrix)
print(pca.singular_values_)
plt.bar(range(30), pca.mean_[:30])
plt.show()

# labelled_words = {**{ word: True for word in positive_words }, **{ word: False for word in negative_words }}
# res = test_dsv_classifier(labelled_words, word_vectors, dsv)
# print(res)

# plot_pca_bar_chart(positive_matrix)
