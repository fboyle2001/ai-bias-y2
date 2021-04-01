import pandas as pd
import numpy as np
from scipy.stats import johnsonsu

def load_and_clean():
    # Load the dataset from the csv
    # It was originally a .mdb file which I converted to .xlsx and then to .csv
    df = pd.read_csv("jee2009.csv")
    # Remove the columns that aren't related to the sensitive attribute
    df = df.drop(["NAME", "category", "sub_category", "PIN_RES", "PARENT_NAM", "math", "phys", "chem"], axis=1)
    # Rename the registration ID column and rename mark to utility for clarity with the terms in the paper
    df = df.rename(columns = { "REGST_NO": "id", "mark": "utility" })
    # It is easier to work with booleans rather than strings and since gender is binary in this dataset
    # we can make it into a boolean representing if they are male (or female if it is false)
    df["is_male"] = np.where(df["GENDER"].str.lower() == "m", True, False)
    # Remove the gender column and return the data frame
    df = df.drop("GENDER", axis=1)
    return df

# Remove before submission and replace with load_and_clean
def faster_load():
    return pd.read_csv("cleaned.csv")

# Biased ranking
# k = no of elements, v = position based discount
def unconstrained_ranking(df, male_beta, female_beta, k, v=lambda x: 1 / np.log(1 + x)):
    # First transform to the observed utility
    uncons_df = df.copy()
    # May need to adjust for the negatives?
    uncons_df["observed_utility"] = np.where(uncons_df["is_male"] == True, uncons_df["utility"] * male_beta, uncons_df["utility"] * female_beta)

    # Now sort them by the observed utility and take the top k records
    ranked_df = uncons_df.sort_values(by="observed_utility", ascending=False)[:k]

    total_latent_utility = 0

    # Calculate the latent utility of the ranked records
    # j is the position (formula 1 in paper)
    m, f = 0, 0

    for j, record in enumerate(ranked_df.itertuples()):
        m += record.is_male
        f += not record.is_male
        total_latent_utility += record.utility * v(j + 1)

    print(m, f)

    return total_latent_utility

def constrained_ranking(df, male_beta, female_beta, k, v=lambda x: 1 / np.log(1 + x)):
    # First transform to the observed utility
    uncons_df = df.copy()
    # May need to adjust for the negatives?
    uncons_df["observed_utility"] = np.where(uncons_df["is_male"] == True, uncons_df["utility"] * male_beta, uncons_df["utility"] * female_beta)


# df = load_and_clean()
# df.to_csv("cleaned.csv", index=False)
# df = faster_load()

male_scores = [int(x) for x in johnsonsu.rvs(-1.3358254338685507, 1.228621987785165, -16.10471198333935, 25.658144591068066, size=100)]
female_scores = [int(x) for x in johnsonsu.rvs(-1.1504808824385124, 1.3649066883190795, -12.879957294149737, 27.482272133428403, size=100)]

male_df = pd.DataFrame(data={
    "is_male": True,
    "utility": male_scores
})

female_df = pd.DataFrame(data={
    "is_male": False,
    "utility": female_scores
})

df = pd.concat([male_df, female_df])

uncons_ranking = unconstrained_ranking(df=df, male_beta=1, female_beta=0.25, k=100)
print(uncons_ranking)
