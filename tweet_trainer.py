from gensim.models.word2vec import Word2Vec, KeyedVectors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from scipy.stats.stats import pearsonr
from matplotlib import pyplot as plt
import time

def text_to_vector(word_vectors, text):
    vec = np.zeros(shape=(300,))
    words = 0

    for word in text.split(" "):
        word = word.strip().lower()

        if word not in word_vectors:
            continue

        words += 1
        vec += word_vectors.get_vector(word)

    if words == 0:
        return None

    return vec / words

def csv_to_vectorised_df(word_vectors, path):
    df = pd.read_csv(path)
    tweet_vectors_with_valence = []

    for _, row in df.iterrows():
        tweet_vector = text_to_vector(word_vectors, row["tweet"])

        if tweet_vector is None:
            continue

        # print(tweet_vector.shape)
        # print(tweet_vector)
        v = np.array([row["valence"]])
        # print(v.shape)
        # print(v)

        w_val = np.concatenate([tweet_vector, v])

        tweet_vectors_with_valence.append(w_val)

    tweet_valence_mat = np.vstack(tweet_vectors_with_valence)
    cols = [f"tf_vc_{i}" for i in range(300)] + ["valence"]
    vectorised_df = pd.DataFrame(tweet_valence_mat, columns=cols)

    # df = df.drop(["id", "tweet_vector", "tweet"], axis=1)
    #
    unlabelled = vectorised_df.drop("valence", axis=1)
    labels = vectorised_df["valence"].copy()
    return unlabelled, labels

    return w, out

def main(debias=False):
    word_vectors = KeyedVectors.load("pretrained-word2vec-keyedvectors.kv")
    # Normalise the vectors
    # CITATION NEEDED this is from a GitHub
    word_vectors.vectors /= np.linalg.norm(word_vectors.vectors, axis=1)[:, np.newaxis]
    std_scaler = StandardScaler(with_std=False, with_mean=True)
    scaled = std_scaler.fit_transform(word_vectors.vectors)
    word_vectors.vectors = scaled

    if debias:
        print("Debiasing")
        lowest = np.loadtxt("./weights/2021-04-05T21-11-28.176502_BEST_weights_a0.5_i40000_s1828.txt")
        vs = []

        st = time.time()

        for v in word_vectors.vectors:
            vs.append(v - lowest * lowest.T * v)

        et = time.time()

        print(f"Debiased took {et-st} seconds")

        word_vectors.vectors = np.array(vs)

        print("Conv to np array")

    # Need to convert the tweets to vectors
    # Then convert the vectors to features

    training_unlabelled, training_labels = csv_to_vectorised_df(word_vectors, "./SemReady/valence_training_set.csv")
    test_unlabelled, test_labels = csv_to_vectorised_df(word_vectors, "./SemReady/valence_test_set.csv")

    print(test_unlabelled.info())

    print("Training SKLEARN")
    #
    model = SVR() # might need to setup a pipeline
    model.fit(training_unlabelled, training_labels)

    print("Trained SKLEARN")
    #
    predictions = model.predict(test_unlabelled)

    print(predictions.shape)
    print(test_labels.shape)

    print(pearsonr(test_labels, predictions))
    print(pearsonr(predictions, test_labels))

    plt.scatter(x=test_labels, y=predictions)
    m, b = np.polyfit(test_labels, predictions, 1)
    #plt.plot(test_labels, predictions)
    plt.plot(test_labels, m*test_labels + b, color="orange")
    plt.show()

main(debias=False)
