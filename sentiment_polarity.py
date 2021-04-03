from gensim.models.word2vec import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time

# Dictionary of files that may be used throughout the program
FILE_PATHS = {
    "pretrained-w2v-google-news": "pretrained-word2vec-keyedvectors.kv",
    "positive_words": "./opinion-lexicon-English/positive-words.txt",
    "negative_words": "./opinion-lexicon-English/negative-words.txt"
}

"""
Loads a file of words from the Sentiment Lexicon (CITATION NEEDED)

word_vectors - The word2vec pre-trained model
filepath - The location of the lexicon file
"""
def load_lexicon(word_vectors, filepath):
    lexicon = set()

    with open(filepath, "r") as file:
        for line in file.readlines():
            # Lines beginning with ; are comments
            if line.startswith(";"):
                continue

            # Strip any whitespace i.e. \n
            cleaned = line.strip()

            # If the line was entirely whitespace or is blank then discard it
            if len(cleaned) == 0:
                continue

            # I want to consider different variation of the words casing
            case_candidates = [cleaned.lower(), cleaned.title(), cleaned.upper()]

            for case_candidate in case_candidates:
                # I also need to consider the different ways that hyphens can be handled
                # I try 3 different possibilities here
                candidates = [
                    case_candidate.replace("-", ""),
                    case_candidate.replace("-", "_"),
                    case_candidate.replace("-", " "),
                ]

                # I then check if any of the candidates are in the trained word2vec model
                # If so I can use this in the lexicon
                for candidate in candidates:
                    if candidate not in word_vectors:
                        continue

                    # Using a set will prevent duplicates
                    lexicon.add(candidate)

    return lexicon

"""
Produce a matrix consisting of the vectors retrieved from the word2vec model
for the words provided.

Each row is a sample and each column is a feature.

word_vectors - The word2vec pre-trained model
lexicon - A list of words
"""
def construct_w2v_matrix(word_vectors, lexicon):
    rows = []

    for word in lexicon:
        # If it not in the model then we won't be able to get the vector
        if word not in word_vectors:
            continue

        # Put the vector in as a row
        rows.append(word_vectors.get_vector(word))

    # Convert to a numpy array so we can use it with PCA
    return np.array(rows)

"""
Uses sklearn to find the principal component axis

matrix - The matrix of vectors to do PCA on
name - File name for outputting the bar chart (if display_chart=True)
n_components - Number of components to find (and plot) in PCA
display_chart - Display (and save) the bar chart of the top n_components explained variance ratios
"""
def principal_component_analysis(matrix, name="unknown", n_components=30, display_chart=False):
    pca = PCA(n_components=n_components)
    pca.fit(matrix)

    if display_chart:
        plt.bar(range(n_components), pca.explained_variance_ratio_[:n_components])
        #CHANGE BEFORE SUBMISSION TO DEF DIRECTORY
        filename = f"./pca_charts/{datetime.now().isoformat().replace(':', '-')}_pca_{name}.png"
        plt.savefig(filename)
        plt.show()

    return pca.components_[0]

"""
Tests the classification based on the scalar projection on to the directional sentiment vector

positive_lexicon - Positive sentiment words
negative_lexicon - Negative sentiment words
word_vectors - The word2vec pre-trained model
dsv - Directional Sentiment Vector
"""
def test_dsv_classification(positive_lexicon, negative_lexicon, word_vectors, dsv):
    # I want to track the total predictions, correct predictions, true positives and
    # false positives (for accuracy and precision)
    total, correct, tp, fp = 0, 0, 0, 0
    # Convert the two arrays to a single dictionary labelling each words with whether
    # they are positive or negative in sentiment (True = positive, False = negative)
    labelled_words = {**{ word: True for word in positive_lexicon }, **{ word: False for word in negative_lexicon }}

    for word in labelled_words:
        # Only interested in the lowercase versions for this
        if word.lower() not in word_vectors:
            continue

        # y_star is the real label
        y_star = labelled_words[word]
        # Scalar projection of the word vector on to the directional sentiment vector
        scalar_projection = np.dot(dsv, word_vectors.get_vector(word.lower()))
        # The sign determines the predicted class (True = positive, False = negative)
        y_star_hat = scalar_projection >= 0

        total += 1

        # Correct
        if y_star_hat == y_star:
            correct += 1
            # True Positive
            if y_star_hat == True:
                tp += 1
        else:
            # False positive
            if y_star_hat == True:
                fp += 1

    accuracy = correct / total * 100
    precision = tp / (tp + fp) * 100

    return { "accuracy": accuracy, "precision": precision }

def test_basic_nationality_classification(word_vectors, dsv):
    identities = ["American", "Mexican", "German", "Italian", "French", "Polish"]
    identity_scores = dict()

    for identity in identities:
        identity_scores[identity] = np.dot(dsv, word_vectors.get_vector(identity.lower()))

    return identity_scores

class StochasticAdversialGradientDescent:
    def __init__(self, dsv, alpha, lr):
        self.dsv = dsv
        self.alpha = alpha
        self.lr = lr
        self.weights = None
        self.adv_weights = None
        self.weights_grad = None
        self.adv_weights_grad = None

    # ys are the sentiment polarity so we want them to go to 0
    def update_gradient(self, W, instances):
        m, f = instances.shape

        # print(instances.shape)

        partials = []

        for j in range(f):
            partial = 0

            for i in range(m):
                partial += np.dot(W, instances[i]) * instances[i][j]

            partials.append(partial)

        partials = np.array(partials, dtype=np.float64)
        partials /= m
        return partials

    def update_weights(self, W, instances, lr):
        # different alpha to the adversary weight
        new_weights = []
        m, f = instances.shape
        # print("m", m)
        grad = self.update_gradient(W, instances)

        for j in range(f):
            new_weights.append(W[j] - lr * grad[j])

        return grad, np.array(new_weights)

    def fit(self, training_data, iters=1000):
        stime = time.time()

        def L_a(adversary_weights, y, y_hat):
            z = np.dot(self.dsv, y)
            z_hat = np.dot(adversary_weights, y_hat)

            return (z - z_hat) ** 2

        def L_p(weights, y):
            w_dp = np.dot(weights, weights)
            scaled_y = w_dp * y
            norm = np.dot(scaled_y, scaled_y)
            return norm

        # We don't have labels as such but rather we want to debias the input
        print(training_data.shape)
        samples, features = training_data.shape
        # A vector
        weights = np.random.default_rng().uniform(low=-1, high=1, size=(features,))
        weights = weights / np.linalg.norm(weights)
        adv_weights = np.random.default_rng().uniform(low=-1, high=1, size=(features,))
        adv_weights = adv_weights / np.linalg.norm(adv_weights)
        alpha = self.alpha

        weights_grad = None
        adv_weights_grad = None

        min_obj = None
        min_obj_vector = None
        min_wg, min_w, min_awg, min_aw = 0, 0, 0, 0

        # self.update_gradient(weights, training_data)

        print(samples, features, weights.shape)

        for it in range(iters):
            lr = np.exp(-5 / iters * it)
            #lr = 0.2 * np.exp(-3 / iters * it)
            #print("It", it)
            u_weights_grad, u_weights = self.update_weights(weights, training_data, lr)
            u_adv_weights_grad, u_adv_weights = self.update_weights(adv_weights, training_data, lr)
            u_obj_vector = np.zeros(weights.shape)

            for y in training_data:
                Lp = L_p(u_weights, y)
                first_term = Lp * u_weights_grad

                y_hat = y - np.dot(u_weights, u_weights) * y
                La = L_a(u_adv_weights, y, y_hat)
                st_scalar_project = np.dot(u_weights_grad * Lp, u_adv_weights_grad * La) / np.sqrt(np.dot(u_adv_weights_grad * La, u_adv_weights_grad * La))
                second_term = st_scalar_project * Lp * u_weights_grad

                third_term = alpha * La * u_adv_weights_grad

                u_obj_vector += first_term + second_term - third_term

            u_obj = np.dot(u_obj_vector, u_obj_vector)

            if min_obj is None or u_obj < min_obj:
                print(it, "new_best", u_obj)
                min_obj = u_obj
                min_obj_vector = u_obj_vector
                min_wg, min_w, min_awg, min_aw = u_weights_grad, u_weights, u_adv_weights_grad, u_adv_weights
            # else:
            #     if np.abs(u_obj - min_obj) > 0.2 * min_obj:
            #         print("resetting")
            #         weights_grad, weights, adv_weights_grad, adv_weights = min_wg, min_w, min_awg, min_aw
            #         continue

            weights_grad, weights, adv_weights_grad, adv_weights = u_weights_grad, u_weights, u_adv_weights_grad, u_adv_weights

        self.weights = min_w
        self.weights_grad = min_wg
        self.adv_weights = min_aw
        self.adv_weights_grad = min_awg

        etime = time.time()
        print("Finished, took", etime - stime, "seconds")

    def debias_vector(self, y):
        scalar = np.dot(self.weights, self.weights)
        print("scale", scalar)
        y_hat = y - scalar * y
        return y_hat

def main(charts=False, validations=False):
    # Load the pre-trained word2vec vectors
    word_vectors = KeyedVectors.load(FILE_PATHS["pretrained-w2v-google-news"])
    # Normalise the vectors
    # CITATION NEEDED this is from a GitHub
    word_vectors.vectors /= np.linalg.norm(word_vectors.vectors, axis=1)[:, np.newaxis]
    scaled = StandardScaler(with_std=False, with_mean=True).fit_transform(word_vectors.vectors)
    word_vectors.vectors = scaled

    # Load the positive and negative lexicons
    positive_lexicon = load_lexicon(word_vectors, FILE_PATHS["positive_words"])
    negative_lexicon = load_lexicon(word_vectors, FILE_PATHS["negative_words"])

    # Convert them matrix form
    positive_matrix = construct_w2v_matrix(word_vectors, positive_lexicon)
    negative_matrix = construct_w2v_matrix(word_vectors, negative_lexicon)

    # Find the most significant principal component axis for both matrices
    # May also show the PCA chart depending on the parameters
    positive_sig_comp = principal_component_analysis(positive_matrix, name="positive_post_norm_no_sd_ss", display_chart=charts)
    negative_sig_comp = principal_component_analysis(negative_matrix, name="negative_post_norm_no_sd_ss", display_chart=charts)

    # The directional sentiment vector is defined to be the signed difference between
    # the principal component axis of the positive and negative sentiment matrices
    dsv = negative_sig_comp - positive_sig_comp
    # Normalise the vector as it will be used for vector and scalar projections
    dsv = dsv / np.sqrt(np.dot(dsv, dsv))

    if validations:
        # This is validation that the idea of projecting on to the DSV is sensible
        # Gives 90.3% precision and 88.2% accuracy when used on the training data
        # i.e. the lexicons
        sentiment_test_results = test_dsv_classification(positive_lexicon, negative_lexicon, word_vectors, dsv)
        # The ordering and distance is fairly okay
        # The only problem is that German and Italian are swapped (by a somewhat significant amount)
        nationality_scores = test_basic_nationality_classification(word_vectors, dsv)

        print("Sentiment Lexicon Classification Validation")
        print(sentiment_test_results)
        print()

        print("Nationality Sentiment Validation")
        print(nationality_scores)
        print()

    debias_model = StochasticAdversialGradientDescent(dsv, alpha=0.5, lr=0.01)
    words = [x.lower() for x in ["male", "female", "he", "she"]]
    # replace with construct_w2v_matrix

    rows = []
    td = np.random.default_rng().choice(word_vectors.vectors, size=(1000,))
    print(td.shape)

    #td = construct_w2v_matrix(word_vectors, words)
    debias_model.fit(td, iters=1000)

    print("W", debias_model.weights)
    print("AW", debias_model.adv_weights)

    print("Now try some words...")
    reg_american = word_vectors.get_vector("male")
    near_reg_am = word_vectors.most_similar(positive=[reg_american], topn=10)
    deb_american = debias_model.debias_vector(reg_american)
    near_deb_am = word_vectors.most_similar(positive=[deb_american], topn=10)

    male_before = np.dot(dsv, reg_american)
    male_after = np.dot(dsv, deb_american)
    print("Male bias before", male_before)
    print("Male bias after", male_after)

    print("Top 10 Near Reg Male")
    for x in near_reg_am:
        print(x)

    print()
    print("Top 10 Near Deb Male")
    for x in near_deb_am:
        print(x)



main(charts=False, validations=False)
