import gensim.downloader as api
import gensim.models
from gensim.models.word2vec import Word2Vec, KeyedVectors
import time
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

positive_lexicon_path = "./opinion-lexicon-English/positive-words.txt"
negative_lexicon_path = "./opinion-lexicon-English/negative-words.txt"

def create_lexicon_list(filepaths):
    cleaned_words = []

    for filepath in filepaths:
        with open(filepath, "r") as file:
            for line in file.readlines():
                # The comments in the file start with a ;
                if line.startswith(";"):
                    continue

                # They end with \n so strip any whitespace
                cleaned_word = line.strip()

                if len(cleaned_word) == 0:
                    continue

                cleaned_words.append(cleaned_word)

    return cleaned_words

def construct_w2v_matrix(wvs, words):
    rows = []

    for word in words:
        if word not in wvs:
            continue

        rows.append(wvs.get_vector(word))

    return np.array(rows)

# Load the vectors from word2vec trained on Google News (word2vec-google-news-300)
wvs = KeyedVectors.load("pretrained-word2vec-keyedvectors.kv")
st = StandardScaler().fit_transform(wvs.vectors)
wvs.vectors = st

# Load the positive and negative words from the Sentiment Lexicon
positive_words = create_lexicon_list([positive_lexicon_path])
negative_words = create_lexicon_list([negative_lexicon_path])
labelled_words = {**{ word: True for word in positive_words }, **{ word: False for word in negative_words }}

# Now construct the matrix consisting of all of the positive and negative vectors
positive_matrix = construct_w2v_matrix(wvs, positive_words)
negative_matrix = construct_w2v_matrix(wvs, negative_words)

# Now get the most significant PCA component of the positive word vector matrix
pca_positive = PCA()
pca_positive.fit_transform(positive_matrix)
ms_pca_comp_pos = pca_positive.components_[0]

plt.bar(range(30), pca_positive.explained_variance_[:30])
plt.show()

# Do the same for the negative word vector matrix
pca_negative = PCA()
pca_negative.fit_transform(negative_matrix)
ms_pca_comp_neg = pca_negative.components_[0]

plt.bar(range(30), pca_negative.explained_variance_[:30])
plt.show()

print(ms_pca_comp_neg.shape, ms_pca_comp_pos.shape)

# Now take the signed difference btween these positive and negative components
dsv = ms_pca_comp_pos - ms_pca_comp_neg
# dsv = dsv / np.sqrt(np.dot(dsv, dsv))

W = ["American", "Mexican", "German", "Italian", "French", "Polish", "admire", "awful", "disgusting", "lovely", "British", "English", "Scottish", "Irish", "Welsh"]

for w in W:
    print(w, np.dot(dsv, wvs.get_vector(w.lower())))

# Try project on some names

def test_dsv_classifier(labelled_words, keyed_vectors, k):
    all = 0
    correct = 0

    for word in labelled_words:
        if word not in keyed_vectors:
            continue

        is_positive = labelled_words[word]
        projection = np.dot(k, keyed_vectors[word])
        y_star_hat = projection + 1 > 0

        if y_star_hat == is_positive:
            correct += 1

        all += 1

    return (correct, all, correct / all * 100)

print(test_dsv_classifier(labelled_words, wvs, dsv))
