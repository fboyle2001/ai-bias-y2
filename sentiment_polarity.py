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
    def __init__(self, dsv, alpha=0.5, seed=None):
        self.dsv = dsv
        self.alpha = alpha
        self.seed = seed
        self.weights = None
        self.weight_scalar = None

    """
    Iteratively calculates the partial derivatives for the gradient of a vector field
    """
    def _update_gradient(self, weights, instances):
        # m is the standard variable for the number of instances
        m, feature_count = instances.shape

        # Stores the partial derivatives as they are calculated
        partials = []

        # Apply the partial derivative update formula where J is a cost function
        # ∂J/∂θ_j = 1/m * sum((θ^T . x^(i) - y^(i)) * x^(i)_j)
        # over each instance x^(i) in the training set and for each feature j
        # y^(i) is a label, in this case we want to get to the debiased sentiment
        # i.e. y^(i) = 0 for all i
        for j in range(feature_count):
            partial = 0
            for i in range(m):
                partial += np.dot(weights, instances[i]) * instances[i][j]

            partials.append(partial)

        partials = np.array(partials, dtype=np.float64)
        partials /= m

        return partials

    """
    Updates the weights iteratively using their gradient
    This is the key part of the gradient descent
    """
    def _update_weights(self, weights, instances, learning_rate):
        # We have to update weights simultaneously so we need a separate array
        new_weights = []
        # m is the standard variable for the number of instances
        m, feature_count = instances.shape
        grad = self._update_gradient(weights, instances)

        for j in range(feature_count):
            # Use the parameter update formula
            # θ_j = θ_j - α * ∂J/∂θ_j
            new_weights.append(weights[j] - learning_rate * grad[j])

        return grad, np.array(new_weights)

    """
    The loss function for the adversarial objective (retain meaning)
    """
    def _adversary_loss(self, weights, y, y_hat):
        # These are just scalars
        # Formulas come from the research paper
        z = np.dot(self.dsv, y)
        z_hat = np.dot(weights, y_hat)
        return (z - z_hat) ** 2

    """
    The loss function for the standard objective (debiaser)
    """
    def _standard_loss(self, weights, y):
        # Dot product is equivalent to ww^T = w.w
        w_norm_sq = np.dot(weights, weights)
        scaled_y = w_norm_sq * y
        sq_norm = np.dot(scaled_y, scaled_y)
        return sq_norm

    """
    This function uses Gradient Descent to fit the training data in order to determine
    the weights to debias a vector
    """
    def train(self, training_instances, iterations=100, verbose=True, batch_size=1000):
        # m is the standard variable for the number of instances
        m, feature_count = training_instances.shape
        # Initialise random weights
        # All of the vectors have been normalised so normalise the weights too
        weights = np.random.default_rng(seed=self.seed).uniform(low=-1, high=1, size=(feature_count,))
        weights = weights / np.sqrt(np.dot(weights, weights))

        # We also need random adversarial weights
        adv_weights = np.random.default_rng(seed=self.seed).uniform(low=-1, high=1, size=(feature_count,))
        adv_weights = weights / np.sqrt(np.dot(weights, weights))

        # We will need to keep track of the minimised objective so we can get the best weights
        min_obj = None
        min_weights = None

        for iteration in range(iterations):
            # The learning_rate decreases later in as we fine tune the model
            # This formula was found by experimentation and plotting graphs on Desmos
            # I wanted the learning_rate to be high at the start and then rapidly decrease
            learning_rate = np.exp(-5 / iterations * iteration)

            # Debugging information
            if verbose:
                print("Iteration:", iteration)
                print("Learning Rate:", learning_rate)

            batch_instances = np.random.default_rng().choice(training_instances, size=(1000,))

            # First get the updated weights and their gradients
            u_weights_grad, u_weights = self._update_weights(weights, batch_instances, learning_rate)
            u_adv_weights_grad, u_adv_weights = self._update_weights(adv_weights, batch_instances, learning_rate)

            # This is the objective that we want to minimise, it will be iteratively set
            u_obj_vector = np.zeros(weights.shape)
            # might need to seed

            # Then we need to calculate the value of the objective function
            # Please refer to the report for the exact formula because it is quite complicated to
            # fit in a comment!
            # Variables names are chosen to reflect what they represent in the original paper
            for y in batch_instances:
                # The first term is a weighted gradient of the standard weights
                Lp = self._standard_loss(u_weights, y)
                first_term = Lp * u_weights_grad

                # The second term is a weighted gradient of standard weights but with
                # the adversarial term having an impact on it to tune the model wrt to both objectives
                y_hat = y - np.dot(u_weights, u_weights) * y
                La = self._adversary_loss(u_adv_weights, y, y_hat)
                scalar_projection = np.dot(u_weights_grad * Lp, u_adv_weights_grad * La)
                # Normalise against the vector we are projecting on to
                scalar_projection /= np.sqrt(np.dot(u_adv_weights_grad * La, u_adv_weights_grad * La))
                second_term = scalar_projection * Lp * u_weights_grad

                # The final term is used to force the two opposing objectives to 'hurt' each other
                # And thus create a better result (otherwise the adversarial objective will help the
                # the original objective rather than contend with it)
                third_term = self.alpha * La * u_adv_weights_grad
                u_obj_vector += first_term + second_term - third_term

            # The dot product = || u_obj_vector ||^2 so it is useful for comparing
            # This constitutes the score used to determine if it is the minimum
            u_obj = np.dot(u_obj_vector, u_obj_vector)

            # Debugging information
            if verbose:
                print("Objective Score:", u_obj)

            # If it is better than save it for later
            # We only want the score and the weights themselves
            # The adversarial weight is only used during the training process
            if min_obj is None or u_obj < min_obj:
                min_obj = u_obj
                min_weights = u_weights

                # Debugging information
                if verbose:
                    print("Beat previous score, new best is now", min_obj)

            # Set the weights ready for use in the next iteration
            weights, adv_weights = u_weights, u_adv_weights

        # Save the best weights we have found
        self.weights = min_weights
        self.weight_scalar = np.dot(self.weights, self.weights)

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
        y_hat = y - self.weight_scalar * y
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

    # The original paper stipulates that we train the debiaser on
    debiaser_training_data = np.random.default_rng(seed=seed).choice(word_vectors.vectors, size=(1000,))
    debiasing_model = AdversarialDebiaser(dsv, alpha=alpha, seed=seed)

    if verbose:
        print(f"Starting training of debiaser model with alpha={alpha}, iterations={iterations}, seed={seed}")
        print("Measuring time taken, this could take a while...")

    start_training_time = time.time()
    debiasing_model.train(word_vectors.vectors, iterations=iterations, verbose=verbose)
    end_training_time = time.time()

    if verbose:
        print("Debiaser trained")
        print(f"Debiaser training took {end_training_time - start_training_time} seconds")

    if validations:
        print("Validating the Debiaser")
        print()
        validate_debiasing(word_vectors, debiasing_model, dsv)

    filename = f"./weights/{datetime.now().isoformat().replace(':', '-')}_weights_a{alpha}_i{iterations}_s{seed}.txt"
    np.savetxt(filename, debiasing_model.weights)

main(charts=False, validations=True, verbose=True, seed=1828, iterations=100, alpha=0.5)
