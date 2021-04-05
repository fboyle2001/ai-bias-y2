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
    "pretrained-glove": "pretrained-glove-keyedvectors.kv",
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

"""
Tests some national identities to measure their bias sentiment

word_vectors - The word2vec pre-trained model
dsv - Directional Sentiment Vector
"""
def test_basic_nationality_classification(word_vectors, dsv):
    identities = ["American", "Mexican", "German", "Italian", "French", "Polish"]
    identity_scores = dict()

    for identity in identities:
        identity_scores[identity] = np.dot(dsv, word_vectors.get_vector(identity.lower()))

    return identity_scores

class AdamOptimiser:
    def __init__(self, size, theta_nought, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=10**(-8)):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.theta_t = theta_nought

        self.t = 0
        self.m_t = np.zeros(shape=(size,))
        self.v_t = np.zeros(shape=(size,))

    def step(self, gradient):
        self.t += 1

        # These are the things at time t - 1
        last_m = self.m_t
        last_v = self.v_t
        last_theta = self.theta_t
        last_grad = gradient

        # Then we process for time t
        m_t = self.beta_1 * last_m + (1 - self.beta_1) * last_grad
        v_t = self.beta_2 * last_v + (1 - self.beta_2) * np.square(last_grad)

        bias_cor_m_t = m_t / (1 - self.beta_1 ** self.t)
        bias_cor_v_t = v_t / (1 - self.beta_2 ** self.t)
        updated_theta = last_theta - self.alpha * bias_cor_m_t / (bias_cor_v_t + self.epsilon)

        self.theta_t = updated_theta
        self.m_t = m_t
        self.v_t = v_t

        return self.theta_t

"""
This class contains everything required to perform Gradient Descent to debias the
word embeddings
"""
class AdversarialDebiaser:
    """
    Create a new instance of this class

    dsv - The directional sentiment vector
    alpha - A parameter used to determine the tradeoff between objectives
    seed - Set the random generator seed (useful for debugging with results that can be replicated)
    """
    def __init__(self, dsv, scaler, seed=None):
        self.dsv = dsv
        self.seed = seed
        self.weights = None
        self.weight_scalar = None
        self.scaler = scaler

    def train(self, training_instances, alpha=0.5, iterations=50, batch_size=1000, verbose=True):
        m, feature_count = training_instances.shape
        rng = np.random.default_rng(seed=self.seed)

        W = rng.uniform(low=-1, high=1, size=(feature_count,))
        # W /= np.sqrt(np.dot(W, W))
        U = rng.uniform(low=-1, high=1, size=(feature_count,))
        # U /= np.sqrt(np.dot(U, U))

        W_opt = AdamOptimiser(300, W)
        U_opt = AdamOptimiser(300, U)

        for epoch in range(iterations):
            print("Epoch", epoch)
            # obj =
            # I need to calculate grad Lp w.r.t W and grad La w.r.t W
            batch = rng.choice(training_instances, size=(batch_size,))
            learning_rate = 0.001

            sum_grad_W = 0
            sum_grad_U = 0
            sum_obj = 0

            for ins_id, y in enumerate(batch):
                # Do this because numpy vectors are difficult
                # w * w^(T) * y as matrices:
                y_w_prod = (W[:, np.newaxis] * W) @ y
                y_hat = y - y_w_prod

                y_w_prod_sum = np.sum(y_w_prod)
                new_grad_Lp_W = 2 / feature_count * y_w_prod_sum * y

                U_dot_y_hat = y_hat.T @ U[:, np.newaxis]
                k_dot_y = np.dot(self.dsv, y)
                La_pre_factor = 2 * (U_dot_y_hat - k_dot_y)

                new_grad_La_W = La_pre_factor * (W * (m + 1) + m - 1)

                normed_La_W = new_grad_La_W / np.linalg.norm(new_grad_La_W)
                scalar_proj = np.dot(new_grad_Lp_W, normed_La_W)
                proj = scalar_proj * normed_La_W

                new_grad_La_U = La_pre_factor * y_hat

                sum_grad_W += new_grad_Lp_W
                sum_grad_U += new_grad_La_U
                sum_obj += new_grad_Lp_W + proj - alpha * new_grad_La_W

            avg_grad_W = sum_grad_W / batch_size
            avg_grad_U = sum_grad_U / batch_size
            avg_obj = sum_obj / batch_size

            print(f"Obj {np.dot(avg_obj, avg_obj)}")

            W = W_opt.step(avg_obj)
            #W /= np.sqrt(np.dot(W, W))
            U = U_opt.step(avg_grad_U)
            #U /= np.sqrt(np.dot(U, U))

            # print("Weights")
            # print(W)
            #
            # print("U WE")
            # print(U)

            #time.sleep(5)

            W_norm = np.sqrt(np.dot(W, W))
            dsv_dot = np.dot(self.dsv, W)
            print(f"W_norm: {W_norm}")
            print(f"Dir: {dsv_dot}")

        self.weights = W
        self.weight_scalar = np.dot(W, W)

    """
    After training we can then debias a vector
    y - The vector to debias
    """
    def debias_vector(self, y):
        if self.weights is None:
            print("Warning: The model has not been trained yet!")
            return y

        # This is the formula that calculates the debiased vector
        # as set out in the original paper
        y_hat = y - self.weights * self.weights.T * y
        return y_hat / np.sqrt((np.dot(y_hat, y_hat)))

"""
Simply calculates the sentiment scalar projection of a word embedding vector

dsv - The directional sentiment vector
vector - The word vector to calculate the scalar projection of
"""
def calculate_sentiment(dsv, vector):
    return np.dot(dsv, vector)

"""
Gets the {topn} nearest neighbours as determine by the word2vec model

word_vectors - The word2vec pre-trained model
vector - The vector of the word to find the neighbours of
topn - The number of neighbours to find
"""
def get_nearest_neighbours(word_vectors, vector, topn=10):
    return word_vectors.most_similar(positive=[vector], topn=topn)

"""
Visual validation of the success of the debiasing effort
Tests it on male and female by default but it can be changed via arguments

word_vectors - The word2vec pre-trained model
debiaser - An AdversarialDebiaser instance (that is trained)
dsv - The directional sentiment vector
words - The words to debias the sentiment of
"""
def validate_debiasing(word_vectors, debiaser, dsv, words=["male", "female"], topn=10):
    for word in words:
        vector = word_vectors.get_vector(word)
        print(f"Analysis of {word}")
        pre_debias_sentiment = calculate_sentiment(dsv, vector)
        print("Pre-debiasing sentiment:", pre_debias_sentiment)
        print(f"{topn} nearest neighbours (pre-debiasing)")

        for entry in get_nearest_neighbours(word_vectors, vector, topn=topn):
            print(entry)

        print()

        debiased_vector = debiaser.debias_vector(vector)
        debias_sentiment = calculate_sentiment(dsv, debiased_vector)
        print("Post-debiasing sentiment:", debias_sentiment)
        relative_sentiment_change = np.abs(pre_debias_sentiment - debias_sentiment) / np.abs(pre_debias_sentiment)
        print(f"Relative debiasing change: {relative_sentiment_change * 100}")
        print(f"{topn} nearest neighbours (post-debiasing)")

        for entry in get_nearest_neighbours(word_vectors, debiased_vector, topn=topn):
            print(entry)

        print("-------------")
        print()

def main(charts=False, validations=False, verbose=False, seed=None, iterations=100, alpha=0.5):
    # Load the pre-trained word2vec vectors
    word_vectors = KeyedVectors.load(FILE_PATHS["pretrained-w2v-google-news"])
    # Normalise the vectors
    # CITATION NEEDED this is from a GitHub
    word_vectors.vectors /= np.linalg.norm(word_vectors.vectors, axis=1)[:, np.newaxis]
    std_scaler = StandardScaler(with_std=False, with_mean=True)
    scaled = std_scaler.fit_transform(word_vectors.vectors)
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
    dsv /= np.sqrt(np.dot(dsv, dsv))
    # Normalise the vector as it will be used for vector and scalar projections

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

    print(f"10 nearest neighbours (male)")

    for entry in get_nearest_neighbours(word_vectors, word_vectors.get_vector("male"), topn=10):
        print(entry)

    print(f"10 nearest neighbours (female)")

    for entry in get_nearest_neighbours(word_vectors, word_vectors.get_vector("female"), topn=10):
        print(entry)

    # The original paper stipulates that we train the debiaser on
    debiaser_training_data = np.random.default_rng(seed=seed).choice(word_vectors.vectors, size=(1000,))
    debiasing_model = AdversarialDebiaser(dsv, seed=seed, scaler=std_scaler)

    if verbose:
        print(f"Starting training of debiaser model with alpha={alpha}, iterations={iterations}, seed={seed}")
        print("Measuring time taken, this could take a while...")

    start_training_time = time.time()
    debiasing_model.train(word_vectors.vectors, alpha=alpha, iterations=1000, verbose=verbose)
    end_training_time = time.time()

    if verbose:
        print("Debiaser trained")
        print(f"Debiaser training took {end_training_time - start_training_time} seconds")

    if validations:
        print("Validating the Debiaser")
        print()
        validate_debiasing(word_vectors, debiasing_model, dsv)

    print("w^T . dsv", np.dot(dsv, debiasing_model.weights))
    print("||w||", np.sqrt(debiasing_model.weight_scalar))

    filename = f"./weights/{datetime.now().isoformat().replace(':', '-')}_weights_a{alpha}_i{iterations}_s{seed}.txt"
    np.savetxt(filename, debiasing_model.weights)

main(charts=False, validations=True, verbose=True, seed=1828, iterations=100, alpha=0.5)
