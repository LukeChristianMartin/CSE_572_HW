# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')
    combine = [train_df, test_df]

    print(train_df.columns.values)


    print(train_df.head().to_string())

    print("\n\n")

    print(train_df.tail().to_string())

    train_df.info()
    print('_' * 40)
    test_df.info()

    print(train_df.describe().to_string())

    print(train_df.describe(include=['O']).to_string())
    print("\n")

    print("\nPclass:")
    print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print("\nSex:")
    print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print("\nSibSp:")
    print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print("\nParch:")
    print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    # g = sns.FacetGrid(train_df, col='Survived')
    # g.map(plt.hist, 'Age', bins=20)


    # grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    # grid.add_legend()

    # grid = sns.FacetGrid(train_df, row='Embarked', aspect=1.6)
    # grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    # grid.add_legend()

    # plt.show()

    print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    print(pd.crosstab(train_df['Title'], train_df['Sex']))

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    print(train_df.head().to_string())

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    print(train_df.shape, test_df.shape)

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    print(train_df.head().to_string())

    # grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    # grid.add_legend()
    #
    # plt.show()

    guess_ages = np.zeros((2, 3))
    print(guess_ages)

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_mean = guess_df.mean()
                age_std = guess_df.std()
                age_guess = rnd.gauss(age_mean, age_std)

                # age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    print(train_df.head())

    train_df['AgeBand'] = pd.cut(train_df['Age'], 8)
    print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 10, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'Age'] = 6
        dataset.loc[dataset['Age'] > 70, 'Age'] = 7



    print(train_df.head().to_string())

    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    print(train_df.head().to_string())


    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False))

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    print(train_df.head())

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10).to_string())

    freq_port = train_df.Embarked.dropna().mode()[0]

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                ascending=False))

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    print(train_df.head().to_string())

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    print(test_df.head().to_string())

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 7)
    print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                ascending=True))

    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.75, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.75) & (dataset['Fare'] <= 8.05), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 8.05) & (dataset['Fare'] <= 12.475), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 12.475) & (dataset['Fare'] <= 19.258), 'Fare'] = 3
        dataset.loc[(dataset['Fare'] > 19.258) & (dataset['Fare'] <= 27.9), 'Fare'] = 4
        dataset.loc[(dataset['Fare'] > 27.9) & (dataset['Fare'] <= 56.929), 'Fare'] = 5
        dataset.loc[dataset['Fare'] > 56.929, 'Fare'] = 6

        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    # exit()
    print(train_df.head(10).to_string())

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    print(X_train.shape, Y_train.shape, X_test.shape)


    correlation_matrix = train_df.corr()

    # Print the correlation matrix
    print(correlation_matrix.to_string())

    # X_train = X_train.drop("Age", axis=1)
    # X_train = X_train.drop("Age*Class", axis=1)
    # X_train = X_train.drop("Fare", axis=1)
    #
    # X_test = X_test.drop("Age", axis=1)
    # X_test = X_test.drop("Age*Class", axis=1)
    # X_test = X_test.drop("Fare", axis=1)

    print(X_train.shape, Y_train.shape, X_test.shape)

    # exit()

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print(acc_log)

    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

    print(coeff_df.sort_values(by='Correlation', ascending=False))

    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    print(acc_svc)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    print(acc_knn)

    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    print(acc_gaussian)

    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    print(acc_perceptron)

    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    print(acc_linear_svc)

    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    print(acc_sgd)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print(acc_decision_tree)

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)

    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree]})
    print(models.sort_values(by='Score', ascending=False))





