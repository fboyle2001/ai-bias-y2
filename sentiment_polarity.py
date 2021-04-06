from gensim.models.word2vec import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

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

"""
Implementation of the Adam Optimiser for Stochastic Gradient Descent Optimisation
Reference: CITATION NEEDED
"""
class AdamOptimiser:
    """
    Adam has a few parameters, these are defaulted to those recommended by the original paper
    outlining the algorithm. theta_t is the value of the parameters at time t.
    """
    def __init__(self, size, theta_nought, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=10**(-8)):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.theta_t = theta_nought

        self.t = 0
        self.m_t = np.zeros(shape=(size,))
        self.v_t = np.zeros(shape=(size,))

    """
    Computes an optimisation step according to the Adam algorithm
    """
    def step(self, gradient):
        self.t += 1

        # These are the things at time t - 1
        last_m = self.m_t
        last_v = self.v_t
        last_theta = self.theta_t
        last_grad = gradient

        # Calculate the zero-biased moments for time t
        m_t = self.beta_1 * last_m + (1 - self.beta_1) * last_grad
        v_t = self.beta_2 * last_v + (1 - self.beta_2) * np.square(last_grad)

        # Corrects the zero bias as m_t and v_t are initialised as zero vectors
        bias_cor_m_t = m_t / (1 - self.beta_1 ** self.t)
        bias_cor_v_t = v_t / (1 - self.beta_2 ** self.t)

        # Update the parameters using the moments
        updated_theta = last_theta - self.alpha * bias_cor_m_t / (bias_cor_v_t + self.epsilon)

        # Save the values for the next step
        self.theta_t = updated_theta
        self.m_t = m_t
        self.v_t = v_t

        return self.theta_t

"""
Debias the word vectors according to two adversarial objectives designed to reduce sentiment polarity
while also keeping word vectors near to their original meaning
Reference: CITATION NEEDED
"""
class AdversarialDebiaser:
    """
    Create a new instance of this class

    dsv - The directional sentiment vector
    alpha - A parameter used to determine the tradeoff between objectives
    seed - Set the random generator seed (useful for debugging with results that can be replicated)
    """
    def __init__(self, dsv, seed=None):
        self.dsv = dsv
        self.seed = seed
        self.lowest = None

    """
    Trains the debiaser according to the objectives using adversarial debiasing
    This takes ~2 hours to train (uses iterations * batch_size samples) on an Intel i7-9700k
    """
    def train(self, training_instances, alpha=0.5, iterations=40000, batch_size=1000, verbose=True):
        # m is the standard for the number of samples
        m, feature_count = training_instances.shape
        # Seed the random so that I can get reproducible results for testing
        rng = np.random.default_rng(seed=self.seed)

        # Initialise random weights for the start
        # W are the weights for the retention of meaning objective
        W = rng.uniform(low=-1, high=1, size=(feature_count,))
        # U are the weights for the debiasing objective
        U = rng.uniform(low=-1, high=1, size=(feature_count,))
        # These variable names are used to match with the papers objective functions

        # Initialise the Adam Optimisers for each objective
        W_opt = AdamOptimiser(300, W)
        U_opt = AdamOptimiser(300, U)

        # Since this is Stochastic it would be possible to get a worse solution at the end epoch
        # than at some epoch in [0, iterations) so I track the best one to give us the best results
        # at the end
        # Ultimately, we only want W, U is used for training only
        lowest_W = None
        min_obj = None

        # Perform the iterative Mini-Batch Gradient Descent
        for epoch in range(iterations):
            if verbose:
                print(f"Epoch {epoch}")

            # Select a batch from the training instances
            batch = rng.choice(training_instances, size=(batch_size,))

            # I average out the gradients and objectives over the batch to use
            # in the optimiser
            sum_grad_W = 0
            sum_grad_U = 0
            sum_obj = 0

            # Calculate the objective and gradients for each instance of the batch
            for y in batch:
                # Compute ww^(T)*y using matrix multiplication
                y_w_prod = (W[:, np.newaxis] * W) @ y
                # y_hat is the sentiment-debiased word vector
                y_hat = y - y_w_prod

                # Sum W element-wise, this is a scaling factor found via the derivation of the gradient
                y_w_prod_sum = np.sum(y_w_prod)
                # This is the gradient of the loss function Lp with respect to W
                new_grad_Lp_W = 2 / feature_count * y_w_prod_sum * y

                # Now we move on to calculating the gradient of La with respect to W
                # Adversarial sentiment polarity
                U_dot_y_hat = y_hat.T @ U[:, np.newaxis]
                # Sentiment polarity
                k_dot_y = np.dot(self.dsv, y)
                # A scaling factor derived during calculation of the gradient
                La_pre_factor = 2 * (U_dot_y_hat - k_dot_y)

                # The gradient of La w.r.t W
                new_grad_La_W = La_pre_factor * (W * (m + 1) + m - 1)

                #
                normed_La_W = new_grad_La_W / np.linalg.norm(new_grad_La_W)
                scalar_proj = np.dot(new_grad_Lp_W, normed_La_W)
                proj = scalar_proj * normed_La_W

                new_grad_La_U = La_pre_factor * y_hat

                sum_grad_W += new_grad_Lp_W
                sum_grad_U += new_grad_La_U
                sum_obj += new_grad_Lp_W - proj - alpha * new_grad_La_W

            avg_grad_W = sum_grad_W / batch_size
            avg_grad_U = sum_grad_U / batch_size
            avg_obj = sum_obj / batch_size

            obj_score = np.dot(avg_obj, avg_obj)

            if min_obj is None or obj_score < min_obj:
                lowest_W = W
                min_obj = obj_score

                if verbose:
                    print(f"New Min Obj {min_obj}")

            # Let the optimiser determine the new weights
            W = W_opt.step(avg_obj)
            U = U_opt.step(avg_grad_U)

        # This takes a long time so it is worth saving the results!
        filename = f"./weights/{datetime.now().isoformat().replace(':', '-')}_best_weights_a{alpha}_i{iterations}_s{self.seed}.txt"
        np.savetxt(filename, lowest_W)

        if verbose:
            print(f"Saved weights to {filename}")

        # Save the trained lowest weights
        self.lowest = lowest_W

    """
    Load pre-determined weights from a txt file
    This is useful because the training takes 2 hours to run!
    """
    def load_from_file(self, filepath):
        # Let numpy load the file into an array
        file = np.loadtxt(filepath)
        self.lowest = file

    """
    After training we can then debias a vector
    y - The vector to debias
    lowest - Whether to use the lowest weights or the last weights
    """
    def debias_vector(self, y):
        # Predictions can't be made if the model is untrained
        if self.lowest is None:
            print("Warning: The model has not been trained yet! (Lowest)")
            return y

        # The, hopefully, debiased word vector!
        y_hat = y - self.lowest * self.lowest.T * y
        return y_hat

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
def validate_debiasing(word_vectors, debiaser, dsv, words=["male", "female", "frank", "american", "josh", "german", "harry", "tia", "jasmine", "betsy"], topn=10):
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

"""
The SemEval-2018 Tweet dataset came as a tab-delimited file.
This function removes unneeded columns and converts the file to a CSV.
"""
def valence_data_to_csv(filepath, savepath):
    # The data frame with the structure of the columns set out
    df = pd.DataFrame(columns=["id", "tweet", "valence"])

    # Open the file with UTF-8 as it contains emojis
    with open(filepath, "r", encoding="utf-8") as file:
        # Loop over each line
        for i, line in enumerate(file.readlines()):
            # The very first line is the column headings
            if i == 0:
                continue

            # Tab-delimited so split on the tabs
            parts = line.split("\t")

            # parts[0] = ID, parts[1] = Tweet, parts[2] = Affect Dimension (irrelevant for this task), parts[3] = Valence
            # Strip whitespace from the ends to remove \n from the valence column
            df = df.append({ "id": parts[0].strip(), "tweet": parts[1].strip(), "valence": parts[3].strip() }, ignore_index=True)

    # Save it to a CSV without an index since we have the IDs
    df.to_csv(savepath, index=False)

def main(charts=False, validations=False, verbose=False, seed=None, iterations=100, alpha=0.5, weights_location=None, debias=True):
    print("Loading the word2vec vectors...")
    # Load the pre-trained word2vec vectors
    word_vectors = KeyedVectors.load(FILE_PATHS["pretrained-w2v-google-news"])
    print("Loaded the word2vec vectors.")
    print("Processing the vector data. For more in-depth information, set verbose=True in the main function.")
    # Normalise the vectors
    # CITATION NEEDED this is from a GitHub
    word_vectors.vectors /= np.linalg.norm(word_vectors.vectors, axis=1)[:, np.newaxis]
    std_scaler = StandardScaler(with_std=False, with_mean=True)
    scaled = std_scaler.fit_transform(word_vectors.vectors)
    word_vectors.vectors = scaled

    if verbose:
        print("Normalised and scaled the word2vec vectors.")

    # Load the positive and negative lexicons
    positive_lexicon = load_lexicon(word_vectors, FILE_PATHS["positive_words"])
    negative_lexicon = load_lexicon(word_vectors, FILE_PATHS["negative_words"])

    if verbose:
        print("Loaded the positive and negative sentiment lexicons.")

    # Convert them matrix form
    positive_matrix = construct_w2v_matrix(word_vectors, positive_lexicon)
    negative_matrix = construct_w2v_matrix(word_vectors, negative_lexicon)

    if verbose:
        print("Converted the sentiment lexicons to word2vec matrices.")

    # Find the most significant principal component axis for both matrices
    # May also show the PCA chart depending on the parameters
    positive_sig_comp = principal_component_analysis(positive_matrix, name="positive_post_norm_no_sd_ss", display_chart=charts)
    negative_sig_comp = principal_component_analysis(negative_matrix, name="negative_post_norm_no_sd_ss", display_chart=charts)

    if verbose:
        print("Completed principal component analysis on each sentiment lexicon.")

    # The directional sentiment vector is defined to be the signed difference between
    # the principal component axis of the positive and negative sentiment matrices
    # Normalise the vector as it will be used for vector and scalar projections
    dsv = negative_sig_comp - positive_sig_comp
    dsv /= np.sqrt(np.dot(dsv, dsv))

    if verbose:
        print("Calculate the directional sentiment vector.")

    print("Processed vector data.")

    if validations:
        # This is validation that the idea of projecting on to the DSV is sensible
        # Gives 90.3% precision and 88.2% accuracy when used on the training data
        # i.e. the lexicons
        sentiment_test_results = test_dsv_classification(positive_lexicon, negative_lexicon, word_vectors, dsv)
        # The ordering and distance is fairly okay
        # The only problem is that German and Italian are swapped (by a somewhat significant amount)
        nationality_scores = test_basic_nationality_classification(word_vectors, dsv)

        print()
        print("Sentiment Lexicon Classification Validation")
        print(sentiment_test_results)
        print()

        print("Nationality Sentiment Validation")
        print(nationality_scores)
        print()

    # Initialise the debiaser
    debiasing_model = AdversarialDebiaser(dsv, seed=seed)

    if verbose:
        print("Debiaser initialised")

    if weights_location is None:
        print("No weights file was provided; training the debiaser instead. This may take a significant amount of time.")

        if verbose:
            print(f"Starting training of debiaser model with alpha={alpha}, iterations={iterations}, seed={seed}")
            print("Measuring time taken, this could take a while...")

        # Train the debiaser model
        start_training_time = time.time()
        debiasing_model.train(word_vectors.vectors, alpha=alpha, iterations=iterations, verbose=verbose)
        end_training_time = time.time()

        if verbose:
            print(f"Debiaser trained. Took {end_training_time - start_training_time} seconds.")

        if validations:
            print("Validating the Debiaser")
            print()
            validate_debiasing(word_vectors, debiasing_model, dsv)

        filename = f"./weights/{datetime.now().isoformat().replace(':', '-')}_weights_a{alpha}_i{iterations}_s{seed}.txt"
        np.savetxt(filename, debiasing_model.weights)
    else:
        debiasing_model.load_from_file(weights_location)
        print("Loaded pre-determined weights for debiaser. Skipped debiaser training.")

    print("Debiaser ready for use.")
    print()
    print("Next Stage: Downstream Sentiment Valence Regression")

main(charts=False, validations=True, verbose=True, seed=1828, iterations=100, alpha=0.5, weights_location="./weights/2021-04-05T21-11-28.176502_BEST_weights_a0.5_i40000_s1828.txt", debias=False)

# valence_data_to_csv("./SemEval-2018/2018-Valence-reg-En-test-gold.txt", "./SemReady/valence_test_set.csv")
