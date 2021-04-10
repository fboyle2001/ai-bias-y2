"""
Bias AI Project Implementation 2021

This project was written in Python 3.8.2 using the following modules and versions:
- gensim 4.0.1
- scikit-learn 0.23.2
- numpy 1.19.2
- matplotlib 3.3.2
- pandas 1.1.3
- scipy 1.5.3
- time, datetime, re, os (standard libraries included with Python)

Before running this please ensure that the folder structure looks like so:
- chpf93.py
- 2018-Valence-reg-En-test-gold.txt
- 2018-Valence-reg-En-train.txt
- negative-words.txt
- positive-words.txt
- 2021-04-08T12-37-34.168980_weights_a0.5_i40000_s1828.txt
- Equity-Evaluation-Corpus.csv

This program will download the Google News Corpus (~1.8 GB download!)
Configuration options are included at the bottom with the main function
"""

import gensim.downloader as api
from gensim.models.word2vec import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
from sklearn.svm import SVR
import scipy.stats.stats as stats
import re
import os

"""
Loads a file of words from the Sentiment Lexicon

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
    # Use sklearns PCA class rather than implement this directly
    pca = PCA(n_components=n_components)
    pca.fit(matrix)

    if display_chart:
        # Displays and saves the PCA chart
        plt.bar(range(n_components), pca.explained_variance_ratio_[:n_components])
        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance Ratio")
        filename = f"{datetime.now().isoformat().replace(':', '-')}_pca_chart_{name}.png"
        plt.savefig(filename)
        plt.show()

    # components_ is sorted by explained variance so this is the top component
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
        if verbose:
            print("Starting training, this will output the current epoch every 1000 epochs.")

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
            if verbose and epoch % 1000 == 0:
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

                # Calculate the gradients using the formulas in my report
                new_grad_Lp_W = 2 / feature_count * y * np.dot(W, y_w_prod)
                La_pre_factor = 2 * (np.dot(U, y_hat) - np.dot(self.dsv, y))
                new_grad_La_W = -La_pre_factor * np.dot(W, y) * (U + y)
                new_grad_La_U = La_pre_factor * y_hat

                # Calculate the objective function
                normed_La_W = new_grad_La_W / np.linalg.norm(new_grad_La_W)
                scalar_proj = np.dot(new_grad_Lp_W, normed_La_W)
                proj = scalar_proj * normed_La_W

                # Keep track of the sum so they can be averaged
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

            # Let the optimiser determine the new weights
            W = W_opt.step(avg_obj)
            U = U_opt.step(avg_grad_U)

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

"""
Converts a string of text to the average of the vector representations of the words
This is the method laid out in the research paper
"""
def text_to_vector(word_vectors, text):
    # Initialise empty variables to compute the averages
    vector = np.zeros(shape=(300,))
    words = 0

    # Split at the spaces
    for word in text.split(" "):
        word = word.strip().lower()
        # Replace all punctuation
        # word = re.sub('[^\w\d\s]', "", word)

        # I can only get an accurate vector if it is in the word embeddings
        if word not in word_vectors:
            continue

        # Keep track of the total words and the total vector
        words += 1
        vector += word_vectors.get_vector(word)

    # If we didn't find any recognised words return None
    if words == 0:
        return None

    # Otherwise return the averaged vector
    return vector / words

"""
Converts the CSV file of the SemEval-2018 dataset to a data frame
Care must be taken because we convert the tweets to their vector form
Each component of the vector then needs to become a vector
"""
def valence_csv_to_vectorised_df(word_vectors, path):
    # Create the initial data frame from the CSV
    df = pd.read_csv(path)
    # We will compile a matrix representation of the rows
    # This can then be imported into a fresh data frame
    tweet_vectors_with_valence = []

    # Iterate each row in the data frame
    # Not the most efficient but the dataset is small so it is still
    # fairly quick and it is clear what is happening
    for _, row in df.iterrows():
        # Convert the tweet to the vector form
        tweet_vector = text_to_vector(word_vectors, row["tweet"])

        # If the vector form is None (i.e. no recognised words) we get rid of it
        if tweet_vector is None:
            continue

        # Concatenate so we have a 301 dimensional array
        vector_and_valence = np.concatenate([tweet_vector, np.array([row["valence"]])])
        tweet_vectors_with_valence.append(vector_and_valence)

    # Construct the matrix of tweets with their valence scores
    tweet_valence_mat = np.vstack(tweet_vectors_with_valence)
    # Each feature is labelled as tf_vc_{index} (Tweet Feature Vector Component {index})
    cols = [f"tf_vc_{i}" for i in range(300)] + ["valence"]
    # Put it all together in a Pandas data frame for ease of use with the model
    vectorised_df = pd.DataFrame(tweet_valence_mat, columns=cols)

    # Split into unlabelled data and the labels so it can be used for training
    # No need to split it in a ratio as the dataset comes pre-split
    unlabelled = vectorised_df.drop("valence", axis=1)
    labels = vectorised_df["valence"].copy()

    return unlabelled, labels

"""
Load the Equity Evaluation Corpus and process the data
"""
def load_gender_EEC_df(path):
    # Create the initial data frame from the CSV
    df = pd.read_csv(path)

    # I'm interested in the male-female cases
    # These are defined to be those that use
    # See Table 3 https://arxiv.org/pdf/1805.04508.pdf
    female_nouns = ["she", "her", "this woman", "this girl", "my sister", "my daughter",
     "my wife", "my girlfriend", "my mother", "my aunt", "my mom"]
    male_nouns = ["he", "him", "this man", "this boy", "my brother", "my son",
    "my husband", "my boyfriend", "my father", "my uncle", "my dad"]

    # I want to filter out all of the rows that do not have these nouns
    df = df[df["Person"].isin(female_nouns) | df["Person"].isin(male_nouns)]

    # Create a dictionary mapping each noun to an index to create gendered pairs
    # e.g. he/she both have index 0, her/him both have index 1
    noun_phrase_index = { **{x: i for i, x in enumerate(female_nouns)}, **{x: i for i, x in enumerate(male_nouns)} }

    # Map the noun phrases to the index
    df["Person"] = df["Person"].map(noun_phrase_index)

    # Drop columns that I do not need
    # Instead of Emotion we are going to use Emotion word
    df = df.drop(["ID", "Race", "Emotion"], axis=1)

    # Convert the Template, Emotion and Emotion word to categorical integers
    template_cats = df["Template"].astype("category").cat.codes
    df["Template"] = template_cats
    ew_cats = df["Emotion word"].astype("category").cat.codes
    df["Emotion word"] = ew_cats

    return df, noun_phrase_index, template_cats.unique(), ew_cats.unique()

"""
Converts the EEC sentences in the data frame to their vectorised representations
"""
def vectorise_df_sentences(word_vectors, df):
    new_df = df.copy()

    # Initialise blank columns to hold the vector components for each sentence
    for i in range(300):
        new_df[f"tf_vc_{i}"] = None

    # Vectorise each sentence and put the components in the data frame
    for row_index, row in df.iterrows():
        sentence_vector = text_to_vector(word_vectors, row["Sentence"])

        for i in range(len(sentence_vector)):
            new_df.at[row_index, f"tf_vc_{i}"] = sentence_vector[i]

    # Remove the old sentence feature
    new_df = new_df.drop("Sentence", axis=1)
    sentence_vectors = new_df.drop(["Template", "Person", "Gender", "Emotion word"], axis=1)
    other_features = new_df.drop([f"tf_vc_{i}" for i in range(300)], axis=1)

    return sentence_vectors, other_features

"""
The main function that initialises and runs everything.
The arguments are explained at the bottom where this function is called instead.
"""
def main(charts=False, validations=False, verbose=False, seed=None, iterations=100, alpha=0.5, weights_location=None, debias=True):
    print("Loading the word2vec vectors...")
    # Load the pre-trained word2vec vectors
    word_vectors = KeyedVectors.load("pretrained-word2vec-keyedvectors.kv")
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
    positive_lexicon = load_lexicon(word_vectors, "positive-words.txt")
    negative_lexicon = load_lexicon(word_vectors, "negative-words.txt")

    if verbose:
        print("Loaded the positive and negative sentiment lexicons.")

    # Convert them matrix form
    positive_matrix = construct_w2v_matrix(word_vectors, positive_lexicon)
    negative_matrix = construct_w2v_matrix(word_vectors, negative_lexicon)

    if verbose:
        print("Converted the sentiment lexicons to word2vec matrices.")

    # Find the most significant principal component axis for both matrices
    # May also show the PCA chart depending on the parameters
    positive_sig_comp = principal_component_analysis(positive_matrix, name="positive", display_chart=charts)
    negative_sig_comp = principal_component_analysis(negative_matrix, name="negative", display_chart=charts)

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

    # If a weights file has been specified then load it
    # Otherwise we need to train the model
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

        # Save the weights as they take a while to generate!
        filename = f"{datetime.now().isoformat().replace(':', '-')}_weights_a{alpha}_i{iterations}_s{seed}.txt"
        np.savetxt(filename, debiasing_model.lowest)
    else:
        # Load using the specified file
        # It is assumed that the file is correctly formatted as this is simply
        # a step for convenience of testing and debugging
        debiasing_model.load_from_file(weights_location)
        print("Loaded pre-determined weights for debiaser. Skipped debiaser training.")

    print("Debiaser ready for use.")
    print()
    print("Next Stage: Downstream Sentiment Valence Regression")

    if debias:
        print("Debias mode enabled. Before regression the word2vecs need to be debiased.")

        # Now we apply the debiasing to each vector
        debias_start_time = time.time()

        # It is important to do this in place otherwise the program maxs out on RAM!
        for i, vector in enumerate(word_vectors.vectors):
            word_vectors.vectors[i] = debiasing_model.debias_vector(vector)

        debias_end_time = time.time()

        if verbose:
            print(f"Debiasing took {debias_end_time - debias_start_time} seconds.")
            print("Now converting it to a numpy array and setting it in the model.")

        print("Debiasing of word2vecs finished.")
        validate_debiasing(word_vectors, debiasing_model, dsv)
    else:
        print("Debias mode disabled. No need to change word embeddings.")

    if verbose:
        print("Loading and vectorising valence training data set (SemEval-2018 Tweets)")

    # Load the SemEval-2018 training data set
    # We don't actually need the test data set unless we are validating later
    training_unlabelled, training_labels = valence_csv_to_vectorised_df(word_vectors, "valence_training_set.csv")

    if verbose:
        print("Loaded valence training data set")
        print("Showing overview stats:")
        print(training_labels.describe())

    print("Training SVR valence model")

    # Fit the SVR model with the training data only
    model = SVR(kernel="linear")
    model.fit(training_unlabelled, training_labels)

    print("Trained SVR valence model")

    if validations:
        test_unlabelled, test_labels = valence_csv_to_vectorised_df(word_vectors, "valence_test_set.csv")
        print("Predicting using test set")

        valence_predictions = model.predict(test_unlabelled)
        # Use scipy to calculate the Pearson's Correlation Coefficient
        valence_pearsons = stats.pearsonr(test_labels, valence_predictions)

        print("Pearsons for the original paper was 0.42 for biased and 0.43 for debiased")
        print(f"Pearsons: {valence_pearsons}")

        if charts:
            # Plot a scatter of the actual and predicted values
            # Then fit a line of best fit through them
            plt.scatter(x=test_labels, y=valence_predictions)
            valence_m, valence_b = np.polyfit(test_labels, valence_predictions, deg=1)
            plt.plot(test_labels, valence_m * test_labels + valence_b, color="orange")
            plt.show()

    print("Now evaluating on the Equity Evaluation Corpus")

    # Load the sentence vectors and their corresponding information
    # i.e. emotion, template, and gender
    EEC_df, noun_phrase_index, template_cats, ew_cats = load_gender_EEC_df("Equity-Evaluation-Corpus.csv")
    EEC_sentence_vectors, EEC_other_features = vectorise_df_sentences(word_vectors, EEC_df)
    # Predict the valence of each sentence in the EEC
    EEC_sentence_valences = model.predict(EEC_sentence_vectors)

    # Set the valences on the data frame for easy access
    EEC_df["valence"] = EEC_sentence_valences

    # EEC_df.sort_values(by=["Template", "Person", "Emotion word", "Gender"]).to_csv("test.csv")

    # I need the unique values so that I can loop over and compare each gendered sentence
    unique_np_indices = set(noun_phrase_index.values())

    # Now we need to see what the valence difference is between sentences of the same template but for different genders
    m_up_f_down = []
    m_down_f_up = []

    # Loop over every template
    for template in template_cats:
        # And every noun phrase
        for npi in unique_np_indices:
            # Select the records that have this specific template and noun phrase
            template_npi_records = EEC_df[(EEC_df["Template"] == template) & (EEC_df["Person"] == npi)]
            # Not all emotion words are used for each sentence
            # So get those that are used for these specific records
            unique_ews = template_npi_records["Emotion word"].unique()

            # Loop over each emotion word in this set of records
            for ew in unique_ews:
                # Select the specific records for this emotion word (should return 2 records)
                # And split into the male and female record
                diff_records = template_npi_records[template_npi_records["Emotion word"] == ew]
                male_record = diff_records[diff_records["Gender"] == "male"].iloc[0]
                female_record = diff_records[diff_records["Gender"] == "female"].iloc[0]

                # Calculate the difference in the valence
                valence_diff = male_record["valence"] - female_record["valence"]

                # If it is > 0 then it is biased towards men
                # We want the absolute value each time
                if valence_diff > 0:
                    m_up_f_down.append(valence_diff)
                else:
                    m_down_f_up.append(-valence_diff)

    # Now display the results

    if charts:
        print("Plotting the sentence deltas for M up, F down sentiment bias")
        plt.scatter(x=np.arange(len(m_up_f_down)), y=m_up_f_down)
        plt.show()

        print("Plotting the sentence deltas for F up, M down sentiment bias")
        plt.scatter(x=np.arange(len(m_down_f_up)), y=m_down_f_up)
        plt.show()

    print("Total pairs biased towards males:", len(m_up_f_down))

    if len(m_up_f_down) != 0:
        print("Mean sentence delta for M up:", sum(m_up_f_down) / len(m_up_f_down))
        print("Sentence delta variance for M up:", np.var(m_up_f_down))

    print()

    print("Total pairs biased towards females:", len(m_down_f_up))

    if len(m_down_f_up) != 0:
        print("Mean sentence delta for F up:", sum(m_down_f_up) / len(m_down_f_up))
        print("Sentence delta variance for F up:", np.var(m_down_f_up))

"""
Checks that files exist for the operation of this project
Also converts data sets to the correct format
"""
def prep_data_sources(weights_location):
    required_files = [
        "2018-Valence-reg-En-test-gold.txt",
        "2018-Valence-reg-En-train.txt",
        "negative-words.txt",
        "positive-words.txt",
        "Equity-Evaluation-Corpus.csv"
    ]

    if weights_location is not None:
        required_files.append(weights_location)

    # Check files needed exist
    for file in required_files:
        if not os.path.isfile(file):
            print(f"Project requires {file} to run!")
            return

    # Do some file conversions if they do not exist
    if not os.path.isfile("valence_training_set.csv"):
        print("valence_training_set.csv not found. Converting the text file.")
        valence_data_to_csv("2018-Valence-reg-En-train.txt", "valence_training_set.csv")

    if not os.path.isfile("valence_test_set.csv"):
        print("valence_test_set.csv not found. Converting the text file.")
        valence_data_to_csv("2018-Valence-reg-En-test-gold.txt", "valence_test_set.csv")

    # Download via gensim
    if not os.path.isfile("pretrained-word2vec-keyedvectors.kv"):
        print("pretrained-word2vec-keyedvectors.kv not found. Downloading and saving via gensim.")
        print("This is a large ~1.8 GB file so it might take a few minutes.")
        downloaded_wvs = api.load("word2vec-google-news-300")
        downloaded_wvs.save("pretrained-word2vec-keyedvectors.kv")

    print("All file validations successful")

# Set to true to generate charts and scatter plots such as the PCA bar chart
# as well as variance plots of sentence deltas and the scatter valence graph
display_charts = True
# Shows the validation checks throughout the project such as validating that the dsv
# is a suitable concept and showing the effects of debiasing on word similarity
display_validation_checks = True
# Define the seed to use for the random generator. It was useful for getting reproducible
# results when developing this!
random_seed = 1828
# The number of iterations for the debiaser
iterations = 40000
# Hyperparameter for the debiaser
alpha = 0.5
# Instead of running the debiaser everytime you can load a pre-trained weights array
# I have included one in the submission
# weights_file_location = "2021-04-08T12-37-34.168980_weights_a0.5_i40000_s1828.txt"
# Uncomment out the line above and comment out the line below to use it!
weights_file_location = None
# Determines whether or not to use the debiaser on the word embeddings
# useful for comparing before and after debiasing
apply_debiaser = False
# Prints additional information about what is happening
verbose = True

# This function will check that all of the required files are present
# it will also download the Google News Corpus and clean the valence data sets
prep_data_sources(weights_file_location)

# This initialises everything in the project
main(charts=display_charts, validations=display_validation_checks, verbose=verbose, seed=random_seed, iterations=iterations, alpha=alpha, weights_location=weights_file_location, debias=apply_debiaser)
