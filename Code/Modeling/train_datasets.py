import numpy as np
from optparse import OptionParser
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
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
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.externals import joblib
from azureml.core import Run

op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

# Retrieve the run and its context (datasets etc.)
run = Run.get_context()

# Load the input datasets from Azure ML
dataset_train = run.input_datasets['train'].to_pandas_dataframe()
dataset_test = run.input_datasets['test'].to_pandas_dataframe()

# Pre-process df for sklearn
# convert to numpy df
data_train = dataset_train.text.values
data_test = dataset_test.text.values

# save orginal target names
target_names = data_train.target_names

# convert label to int
y_train = dataset_train.target.values
y_test = dataset_test.target.values

# Extracting features from the training data using a sparse vectorizer")
vectorizer = HashingVectorizer(
    stop_words='english',
    alternate_sign=False,
    n_features=op.n_features
)

X_train = vectorizer.transform(data_train.data)

# Extracting features from the test data using the same vectorizer
X_test = vectorizer.transform(data_test.data)

# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()

# # Extracting %d best features by a chi-squared test
# ch2 = SelectKBest(chi2, k=op.select_chi2)
# X_train = ch2.fit_transform(X_train, y_train)
# X_test = ch2.transform(X_test)

# keep selected feature names
# feature_names = [feature_names[i] for i
#                     in ch2.get_support(indices=True)]
# feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


def benchmark(clf, name):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    score = metrics.accuracy_score(y_test, pred)

    child_run = run.child_run(name=name)
    child_run.log("accuracy", float(score))
    model_name = "model" + str(name) + ".pkl"
    filename = "outputs/" + model_name
    joblib.dump(value=clf, filename=filename)
    child_run.upload_file(name=model_name, path_or_stream=filename)

    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if op.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if op.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if op.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]

    child_run.complete()
    return clf_descr, score, train_time, test_time


results = []

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, name))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    name = penalty +  "LinearSVC"
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    name = penalty + "SGDClassifier"
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
name = "Elastic-Net penalty"
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
name ="NearestCentroid (aka Rocchio classifier)"
results.append(benchmark(NearestCentroid()))


# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
name = "Naive Bayes MultinomialNB"
results.append(benchmark(MultinomialNB(alpha=.01)))

name = "Naive Bayes BernoulliNB"
results.append(benchmark(BernoulliNB(alpha=.01)))

name = "Naive Bayes ComplementNB"
results.append(benchmark(ComplementNB(alpha=.1)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
name = "LinearSVC with L1-based feature selection"
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))
