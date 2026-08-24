import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data():

    df = pd.read_csv(
        "data/processed/clean_nfhs_menopause.csv"
    )

    return df


def prepare_data():

    df = load_data()

    # --------------------------------
    # CREATE TARGET
    # --------------------------------

    df["target"] = (
        df["menopause_stage"] == "Perimenopausal"
    ).astype(int)

    # --------------------------------
    # FEATURES
    # --------------------------------

    X = df[
        [
            "age",
            "residence_type",
            "wealth_index",
            "is_pregnant"
        ]
    ]

    y = df["target"]

    # --------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())