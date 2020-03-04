import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]


print("Loading 20 newsgroups dataset for categories:")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

# Extracting features from the training data using a sparse vectorizer
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)

# Extracting features from the test data using the same vectorizer"
X_test = vectorizer.transform(data_test.data)

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


def benchmark(clf, name=""):
    """benchmark classifier performance"""

    # train a model
    print("\nTraining run with algorithm \n{}".format(clf))
    clf.fit(X_train, y_train)

    # evaluate on test set
    pred = clf.predict(X_test)   
    score = ?

    clf_descr = str(clf).split('(')[0]
    print("?  %0.3f" % score)
    return clf_descr, score


# Run benchmark and collect results with multiple classifiers
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
    # run benchmarking function for each
    benchmark(clf, name)


# Run with different regularization techniques
for penalty in ["l2", "l1"]:
    # Train Liblinear model
    name = penalty + "LinearSVC"
    benchmark(
        clf=LinearSVC(
            penalty=penalty,
            dual=False,
            tol=1e-3
        ),
        name=penalty + "LinearSVC"
    )

    # Train SGD model
    benchmark(
        SGDClassifier(
            alpha=.0001,
            max_iter=50,
            penalty=penalty
        ),
        name=penalty + "SGDClassifier"
    )

# Train SGD with Elastic Net penalty
benchmark(
    SGDClassifier(
        alpha=.0001,
        max_iter=50,
        penalty="elasticnet"
    ),
    name="Elastic-Net penalty"
)

# Train NearestCentroid without threshold
benchmark(
    NearestCentroid(),
    name="NearestCentroid (aka Rocchio classifier)"
)

# Train sparse Naive Bayes classifiers
benchmark(
    MultinomialNB(alpha=.01),
    name="Naive Bayes MultinomialNB"
)

benchmark(
    BernoulliNB(alpha=.01),
    name="Naive Bayes BernoulliNB"
)

benchmark(
    ComplementNB(alpha=.1),
    name="Naive Bayes ComplementNB"
)

# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
benchmark(
    Pipeline([
        ('feature_selection',
            SelectFromModel(
                LinearSVC(
                    penalty="l1",
                    dual=False,
                    tol=1e-3
                )
            )),
        ('classification',
            LinearSVC(penalty="l2"))
        ]
    ),
    name="LinearSVC with L1-based feature selection"
)
