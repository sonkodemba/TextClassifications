from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk import accuracy
from nltk import DecisionTreeClassifier
from nltk.classify.svm import SvmClassifier
import sklearn.svm
import random
import nltk
nltk.download('names')


gender = [(n,'Male') for n in names.words('male.txt')] + [(name,'FeMale') for name in names.words('female.txt')]

random.shuffle(gender)

gender_feature = lambda word : {'feature-set':word[-1]}
# The Above function is the same as
#def gender_feature(word):
#determine if a word is Gender or not
#this will return a feature-set or result-set or training-set
#the Last Letter of the Word.


# Next, we use the feature extractor to process the names data, and
# divide the resulting list of feature sets into a training set
# The training set is used to train a new "naive Bayes" classifier.

featuresets = [(gender_feature(name), predictor) for (name, predictor) in gender]
trainingSet, testSet = featuresets[500:], featuresets[0:500]
# print(featuresets)
# # print(trainingSet)
# # print(testSet)



name = input('Enter a Name:')
classifiers = int(input('Enter a Classifier:\n1:Naive Bayes Classifier\n2: Decision Tree\n3: Support Vector machine (SVM)'))
accuracies = []
trainNaiveBaseClassifier = lambda data: NaiveBayesClassifier.train(data)
decisionTreeBaseClassifier = lambda decision: DecisionTreeClassifier.train(decision)
supportVectorMachine = lambda svm: classify.SklearnClassifier(LinearSVC())

if classifiers == 1:

        # naiveClassifier = NaiveBayesClassifier.train(trainingSet)

        naiveBayesClassifier = trainNaiveBaseClassifier(trainingSet)
        n =naiveBayesClassifier.classify(gender_feature(name))
        print('*'*80)
        print('Naive Bays Classifier')
        print('*' * 80)
        print(naiveBayesClassifier.show_most_informative_features())

        #test the accuracy  of the classifier using Naive Bayes
        # Observe that these character names from The Matrix are correctly classified. Although this science fiction movie
        # is set in 2199, it still conforms with our expectations about names and genders. We can systematically evaluate
        # the classifier on a much larger quantity of unseen data:

        #computing Accuracy:
        naive_accuracy = classify.accuracy(naiveBayesClassifier, testSet)
        accuracies.append(('Naive Bayes : ', naive_accuracy))
        print('Naive Accuracy = ', naive_accuracy)
elif classifiers == 2:
    # Decision tree learning is one  of

        # decision_tree = DecisionTreeClassifier.train(trainingSet)
        decisionTree = decisionTreeBaseClassifier(trainingSet)
        classifyDecisionTree =decisionTree.classify(gender_feature(name))
        decision_tree_accuracy = classify.accuracy(decisionTree, testSet)
        accuracies.append(('Decision Tree Accuracy : ', decision_tree_accuracy))
        print('decision Tree Accuracy = ',decision_tree_accuracy,':', decisionTree)
        print('*'*80)

elif classifiers == 3:
    pass
    # print('*'* 80)
    # # Note
    # # nltk.classify.svm was deprecated.For classification based on support vector
    # # machines SVMs use nltk.classify.scikitlearn( or scikit - learn directly).For more details NLTK 3.0 documentation
    #     classifier = classify.SklearnClassifier(LinearSVC())
    #     classifier.train(trainingSet)
    #     c = classifier.classify(gender_feature(name))
    #     accuracy = classify.accuracy(classifier, testSet)
    #     print('Prediction:',name,' = ',c, 'Accuracy : ', accuracy)
    #     print('*'*80)
    #
