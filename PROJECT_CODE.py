import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import cross_val_score


march_madness_file_path = 'cbb.csv'
trials = Trials()
iteration_list = []
accuracy_list = []

def objective(params):
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    iteration_list.append(len(trials))
    accuracy_list.append(score)  # We use negative score since hyperopt minimizes the objective function
    return -score

def plot_columns(df, x_col, y_col, title="Scatter Plot", x_label="X-axis", y_label="Y-axis"):
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.scatter(df[x_col], df[y_col], marker='o', color='b', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def encode_postseason(march_madness_df):

    post_season_mapping = {'NaN':       0,
                           'R68':       1,
                           'R64':       2,
                           'R32':       3,
                           'S16':       4,
                           'E8':        5,
                           'F4':        6,
                           '2ND':       7,
                           'Champions': 8}

    march_madness_df['POSTSEASON_ENCODE'] = march_madness_df['POSTSEASON'].map(post_season_mapping)
    march_madness_df['POSTSEASON_ENCODE'].fillna(0, inplace=True)
    march_madness_df['POSTSEASON_ENCODE'] = march_madness_df['POSTSEASON_ENCODE'].astype(int)
    march_madness_df.drop('POSTSEASON', axis=1, inplace=True)

def encode_seed(march_madness_df):
    march_madness_df['SEED'].fillna(17, inplace=True)

    march_madness_df['SEED_ENCODE'] = march_madness_df['SEED'].astype(int)
    march_madness_df.drop('SEED', axis=1, inplace=True)


def drop_columns_for_pitch_part_1(march_madness_df, list_of_columns_drop):

    for cols in list_of_columns_drop:
        march_madness_df.drop(cols, axis=1, inplace=True)

    return march_madness_df


if __name__ == '__main__':
    mm_df = pd.read_csv(march_madness_file_path)

    encode_postseason(mm_df)
    encode_seed(mm_df)

    mm_df.drop('TEAM', axis=1, inplace=True)
    mm_df.drop('CONF', axis=1, inplace=True)
    mm_df.drop('YEAR', axis=1, inplace=True)


    ############## Create new Varibales for more data ###############
    mm_df["TOR_DIFF"] = mm_df["TORD"] - mm_df["TOR"]

    mm_df["EXTRA_POS"]      = (mm_df["ORB"] + mm_df["TORD"]) - (mm_df["DRB"] + mm_df["TOR"])
    mm_df["3PT_DIFF"]       = mm_df["3P_O"] - mm_df["3P_D"]
    mm_df["2PT_DIFF"]       = mm_df["2P_O"] - mm_df["2P_D"]
    mm_df["FT_DIFF"]        = mm_df["FTR"] - mm_df["FTRD"]
    mm_df["SHOOT_DIFF"]     = (mm_df["FT_DIFF"] * 1) + (mm_df["2PT_DIFF"] * 2) + (mm_df["3PT_DIFF"] * 3)
    ############## Create new Varibales for more data ###############

    # combination = pd.concat([input_data, output_data], axis=1)

    indices_to_remove = mm_df[mm_df['POSTSEASON_ENCODE'] == 0].index.tolist()

    half_count = int(len(indices_to_remove) * 0.85)

    random.seed(85)
    indices_to_remove = random.sample(indices_to_remove, half_count)

    # Remove the selected rows
    mm_df = mm_df.drop(indices_to_remove)

    # Reset the index to ensure it's continuous
    mm_df.reset_index(drop=True, inplace=True)

    count_specific_value = (mm_df['POSTSEASON_ENCODE'] == 0).sum()

    percentage = (count_specific_value / len(mm_df)) * 100

    print(f"The percentage of 0s in the POSTSEASON_ENCODE column is: {percentage:.2f}%")

    covariance_matrix = mm_df.corr()

    print(covariance_matrix.to_string())


    # Size of this list is 133
    cols_to_drop = ["TOR", "TOR_DIFF", "TORD", "3PT_DIFF", "2PT_DIFF", "FT_DIFF", "EFG_O", "EFG_D", "DRB", "FTR",
                    "FTRD", "2P_O", "2P_D", "3P_D", "ADJ_T"]

    drop_columns_for_pitch_part_1(mm_df, cols_to_drop)

    covariance_matrix = mm_df.corr()

    print(covariance_matrix.to_string())

    output_data = mm_df['POSTSEASON_ENCODE']

    mm_df.drop('POSTSEASON_ENCODE', axis=1, inplace=True)
    input_data = mm_df

    # z-score normalization
    z_score_scaler = StandardScaler()
    columns_z_score_norm = ["ORB", "WAB", "ADJOE", "ADJDE", "G", "EXTRA_POS", "SHOOT_DIFF"]
    input_data[columns_z_score_norm] = z_score_scaler.fit_transform(input_data[columns_z_score_norm])

    # z-score normalization
    linear_scaler = MinMaxScaler()
    columns_linear_scaler = ["SEED_ENCODE", "W"]
    input_data[columns_linear_scaler] = linear_scaler.fit_transform(input_data[columns_linear_scaler])

    input_data["3P_O"] = input_data["3P_O"]/100

    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)


    space = {
        'n_estimators': hp.uniformint('n_estimators', 10, 200),
        'max_depth': hp.uniformint('max_depth', 1, 30),
        'min_samples_split': hp.uniformint('min_samples_split', 2, 20),
        'min_samples_leaf': hp.uniformint('min_samples_leaf', 1, 20)
    }

    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)


    plt.plot(iteration_list, accuracy_list, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Bayesian Optimization for Random Forest Hyperparameter Tuning')
    plt.show()

    # Retrain the model with the best parameters on the entire dataset
    best_params_int = {key: int(value) for key, value in best_params.items()}

    best_model = RandomForestClassifier(**best_params_int)
    best_model.fit(X_train, y_train)

    # Evaluate the model with the best parameters on the test set
    test_accuracy = best_model.score(X_test, y_test)

    # Print the best parameters
    print("Best Parameters:", best_params)

    # Print the accuracy on the test set
    print("Test Accuracy:", test_accuracy)





