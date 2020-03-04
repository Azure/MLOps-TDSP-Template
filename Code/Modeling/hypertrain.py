import logging
from optparse import OptionParser
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.externals import joblib
from azureml.core import Run

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
              action="store",
              type=int,
              default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--max_depth",
              type=int, default=10)
op.add_option("--n_estimators",
              type=int, default=100)
op.add_option("--criterion",
              type=str,
              default='gini')
op.add_option("--min_samples_split",
              type=int,
              default=2)


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
print(categories if categories else "all")

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

print("Extracting features from the training data using a sparse vectorizer")
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(data_test.data)


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()


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
    score = metrics.accuracy_score(y_test, pred)

    # log metrics
    run_logger = Run.get_context()
    run_logger.log("accuracy", float(score))

    # save .pkl file
    model_name = "model" + ".pkl"
    filename = "outputs/" + model_name
    joblib.dump(value=clf, filename=filename)
    run_logger.upload_file(name=model_name, path_or_stream=filename)

    print("accuracy:   %0.3f" % score)
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score


results = []

# Select the training hyperparameters.
# Create a dict of hyperparameters from the input flags.
hyperparameters = {
    "max_depth": opts.max_depth,
    "n_estimators": opts.n_estimators,
    "criterion": opts.criterion,
    "min_samples_split": opts.min_samples_split
}

# Select the training hyperparameters.
max_depth = hyperparameters["max_depth"]
n_estimators = hyperparameters["n_estimators"]
criterion = hyperparameters["criterion"]
min_samples_split = hyperparameters["min_samples_split"]


clf = RandomForestClassifier(max_depth=max_depth,
                             n_estimators=n_estimators, criterion=criterion,
                             min_samples_split=min_samples_split)

model = benchmark(clf)
