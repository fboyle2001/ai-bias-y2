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

def create_lexicon_list(wvs, filepaths):
    cleaned_words = []

    for filepath in filepaths:
        skipped = []

        with open(filepath, "r") as file:
            for line in file.readlines():
                # The comments in the file start with a ;
                if line.startswith(";"):
                    continue

                # They end with \n so strip any whitespace
                cleaned_word = line.strip()

                if len(cleaned_word) == 0:
                    continue

                for op in [1, 2, 3]:
                    temp = ""

                    if op == 1:
                        temp = cleaned_word.lower()
                    elif op == 2:
                        temp = cleaned_word.title()
                    elif op == 3:
                        temp = cleaned_word.upper()

                    if temp not in wvs:
                        temp = temp.replace("-", "_")

                        if temp not in wvs:
                            temp = temp.replace("_", " ")

                            if cleaned_word not in wvs:
                                temp = temp.replace(" ", "")

                                if temp not in wvs:
                                    skipped.append(temp)
                                    continue

                    cleaned_words.append(temp)

        t = filepath.split('/')[-1]
        with open(f"skipped_{t}.txt", "w") as f:
            f.write("\n".join(skipped))

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
# st = StandardScaler().fit_transform(wvs.vectors)
# wvs.vectors = st
norms = np.linalg.norm(wvs.vectors, axis=1)

if max(norms) - min(norms) > 0.0001:
    print("Normalising")
    wvs.vectors /= np.linalg.norm(wvs.vectors, axis=1)[:, np.newaxis]

pairs = [('she', 'he'),
('her', 'his'),
('woman', 'man'),
('Mary', 'John'),
('herself', 'himself'),
('daughter', 'son'),
('mother', 'father'),
('gal', 'guy'),
('girl', 'boy'),
('female', 'male')]

def doPCA(pairs, embedding, num_components = 10):
    matrix = []
    for a, b in pairs:
        center = (wvs[a] + wvs[b])/2
        matrix.append(wvs[a] - center)
        matrix.append(wvs[b] - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    plt.bar(range(num_components), pca.explained_variance_ratio_)
    plt.show()
    return pca

doPCA(pairs, wvs)

# Load the positive and negative words from the Sentiment Lexicon
positive_words = create_lexicon_list(wvs, [positive_lexicon_path])
negative_words = create_lexicon_list(wvs, [negative_lexicon_path])
labelled_words = {**{ word: True for word in positive_words }, **{ word: False for word in negative_words }}

# Now construct the matrix consisting of all of the positive and negative vectors
positive_matrix = construct_w2v_matrix(wvs, positive_words)
negative_matrix = construct_w2v_matrix(wvs, negative_words)

# Now get the most significant PCA component of the positive word vector matrix
ncomps = 30
pca_positive = PCA(n_components=ncomps)
pca_positive.fit_transform(positive_matrix)
ms_pca_comp_pos = pca_positive.components_[0]

plt.bar(range(ncomps), pca_positive.explained_variance_ratio_[:ncomps])
plt.show()

# Do the same for the negative word vector matrix
pca_negative = PCA(n_components=ncomps)
pca_negative.fit_transform(negative_matrix)
ms_pca_comp_neg = pca_negative.components_[0]

plt.bar(range(ncomps), pca_negative.explained_variance_ratio_[:ncomps])
plt.show()

print(ms_pca_comp_neg.shape, ms_pca_comp_pos.shape)

# Now take the signed difference btween these positive and negative components
dsv =  ms_pca_comp_neg - ms_pca_comp_pos
# dsv = dsv / np.sqrt(np.dot(dsv, dsv))

# W = ["American", "Mexican", "German", "Italian", "French", "Polish", "admire", "awful", "disgusting", "lovely", "British", "English", "Scottish", "Irish", "Welsh"]
#
# for w in W:
#     t = w
#     if w not in wvs:
#         t = w.lower()
#         if w not in wvs:
#             t = w.upper()
#             if w not in wvs:
#                 t = w.title()
#                 if w not in wvs:
#                     print(w, "Missing")
#     print(t, np.dot(dsv, wvs.get_vector(t)))

# Try project on some names

def test_dsv_classifier(labelled_words, keyed_vectors, k):
    all = 0
    correct = 0
    tp, fp = 0, 0

    for word in labelled_words:
        if word.lower() not in keyed_vectors:
            continue

        is_positive = labelled_words[word]
        projection = np.dot(k, keyed_vectors.get_vector(word.lower()))
        y_star_hat = projection >= 0

        if y_star_hat == is_positive:
            if y_star_hat == True:
                tp += 1

            correct += 1
        elif y_star_hat == True:
            fp += 1

        all += 1

    return (correct, all, tp / (fp + tp), (correct  / all * 100))

print(test_dsv_classifier(labelled_words, wvs, dsv))
