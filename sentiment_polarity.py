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

    for word in word_list:
        vector = None

        # Not all of the words will be in the corpus so skip them
        try:
            vector = keyed_vectors.get_vector(word)
        except KeyError:
            continue

        if vector is None:
            continue

        matrix.append(vector)

    # array will make them the rows so we need to transpose to get the vectors
    # as the columns
    return np.transpose(np.array(matrix))

def prepare_lexicon(keyed_vectors, filepath):
    words = create_lexicon_list(filepath)
    matrix = prepare_vector_matrix(keyed_vectors, words)

    return words, matrix

def get_principal_variances(matrix, n=30):
    pca = PCA(n_components=n)
    pca.fit(matrix)
    return pca.explained_variance_

def get_principal_components(matrix, n=1):
    pca = PCA(n_components=n)
    pca.fit(matrix)
    return pca.components_

def plot_pca_bar_chart(matrix, n=30, file=None, show=True):
    principal_exp_variances = get_principal_variances(matrix, n)
    plt.bar(np.arange(0, n), principal_exp_variances * 10)

    if file is not None:
        plt.savefig(file)

    if show:
        plt.show()

def project_vector():
    pass

# keyed_vectors = api.load("word2vec-google-news-300")
# keyed_vectors.save("pretrained-word2vec-keyedvectors.kv")
# keyed_vectors = api.load("glove-wiki-gigaword-300")
# keyed_vectors.save("pretrained-glove-keyedvectors.kv")

word_vectors = KeyedVectors.load("pretrained-word2vec-keyedvectors.kv")
positive_words, positive_matrix = prepare_lexicon(word_vectors, positive_lexicon_path)
negative_words, negative_matrix = prepare_lexicon(word_vectors, negative_lexicon_path)

X = word_vectors.vectors
pca = PCA(n_components=2)
result = pca.fit_transform(X)
print(pca.components_.shape)
plt.bar(np.arange(0, 2), pca.explained_variance_)
plt.show()

# result = pca.fit_transform()

# pos_pca = PCA()
# pos_pca.fit(positive_matrix)
# first_pos_pc = pos_pca.components_[:,0]
#
# neg_pca = PCA()
# neg_pca.fit(negative_matrix)
# first_neg_pc = neg_pca.components_[:,0]
#
# directional_sentiment_vector = first_pos_pc - first_neg_pc
# normalised_dsv = directional_sentiment_vector / np.linalg.norm(directional_sentiment_vector)
#
# polish_projection_scalar = np.dot(word_vectors["Polish"], normalised_dsv)
# american_projection_scalar = np.dot(word_vectors["American"], normalised_dsv)
# french_projection_scalar = np.dot(word_vectors["French"], normalised_dsv)
#
# print(polish_projection_scalar)
# print(american_projection_scalar)
# print(french_projection_scalar)

# plot_pca_bar_chart(positive_matrix, file="fig.png")

# pos_pcs = get_principal_components(positive_matrix)
# neg_pcs = get_principal_components(negative_matrix)
# plot_pca_bar_chart(positive_matrix)
# #
# # print(pos_pcs)
# # print(neg_pcs)
# print(pos_pcs.shape)
# print(neg_pcs.shape)

# directional_sentiment_vector = pos_pcs - neg_pcs
# print(directional_sentiment_vector.shape)
# print(directional_sentiment_vector)
# plt.bar(np.arange(0, 30), directional_sentiment_vector)
# plt.show()
