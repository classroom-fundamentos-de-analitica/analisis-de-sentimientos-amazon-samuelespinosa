import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def pregunta_01():
    df = pd.read_csv('amazon_cells_labelled.tsv', sep='\t', header=None, names=['msg', 'lbl'])
    df_tagged = df[df['lbl'].notnull()]
    df_untagged = df[df['lbl'].isnull()]
    x_tagged = df_tagged['msg']
    y_tagged = df_tagged['lbl']
    x_untagged = df_untagged['msg']
    y_untagged = df_untagged['lbl']
    return x_tagged, y_tagged, x_untagged, y_untagged


def pregunta_02():
    x_tagged, y_tagged, _, _ = pregunta_01()
    x_train, x_test, y_train, y_test = train_test_split(x_tagged, y_tagged, test_size=0.1, random_state=12345)
    return x_train, x_test, y_train, y_test


def pregunta_03():
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()
    return lambda x: (stemmer.stem(w) for w in analyzer(x))


def pregunta_04():
    x_train, _, y_train, _ = pregunta_02()
    analyzer = pregunta_03()
    countVectorizer = CountVectorizer(
        analyzer=analyzer,
        lowercase=True,
        stop_words=None,
        token_pattern=r'\b[a-zA-Z]+\b',
        binary=True,
        max_df=1.0,
        min_df=5
    )
    pipeline = Pipeline([
        ("vectorizer", countVectorizer),
        ("model", BernoulliNB()),
    ])
    param_grid = {
        "model__alpha": np.linspace(0.1, 1.0, 10),
    }
    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        refit=True,
        return_train_score=True,
    )
    gridSearchCV.fit(x_train, y_train)
    return gridSearchCV


def pregunta_05():
    gridSearchCV = pregunta_04()
    X_train, X_test, y_train, y_test = pregunta_02()
    cfm_train = confusion_matrix(y_true=y_train, y_pred=gridSearchCV.predict(X_train))
    cfm_test = confusion_matrix(y_true=y_test, y_pred=gridSearchCV.predict(X_test))
    return cfm_train, cfm_test


def pregunta_06():
    gridSearchCV = pregunta_04()
    _, _, X_untagged, _ = pregunta_01()
    y_untagged_pred = gridSearchCV.predict(X_untagged)
    return y_untagged_pred

