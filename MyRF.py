import pandas as pd # load csv's (pd.read_csv)
import numpy as np # math (lin. algebra)
import sklearn as skl # machine learning
from sklearn.ensemble import RandomForestClassifier

# Visualisation
import matplotlib.pyplot as plt # plot the data
import seaborn as sns # data visualisation
sns.set(color_codes=True)
#% matplotlib inline

def input_and_visual(trainname, testname):  # load data as Pandas.DataFrame
    train_df = pd.read_csv(trainname)
    train_data = train_df.values  #  .value could abstract the data from targeted value

    test_df = pd.read_csv(testname)
    test_data = test_df.values

    plt.figure(figsize=(12,8))
    sns.countplot(x='label', data=train_df)  #  seaborn could be used for draw statistic picture @https://blog.csdn.net/bj_109/article/details/86516598
    plt.title('Distribution of Numbers')
    plt.xlabel('Numbers');
    #plt.show()
# Holdout ( 2/3 to 1/3 )
    num_features = train_data.shape[0] # number of features
    print("Number of all features: \t\t", num_features)
    split = int(num_features * 2/3)

    train = train_data[:split]
    test = train_data[split:]

    print("Number of features used for training: \t", len(train),
          "\nNumber of features used for testing: \t", len(test))
    return train, train_data, test, test_data

# Classifier
def buildRandomForest(tree_number, train, train_data, test, test_data):

    clf = RandomForestClassifier(n_estimators=tree_number) # 100 trees

# train model / ravel to flatten the array structure from [[]] to []
    model = clf.fit(train[:,1:], train[:,0].ravel())

# evaluate on testdata
    output = model.predict(test[:,1:])
    trainput = model.predict(train[:, 1:])

# calculate accuracy
    test_acc = np.mean(output == test[:,0].ravel()) * 100 # calculate accuracy
    train_acc = np.mean(trainput == train[:, 0].ravel()) * 100
    print("The accuracy in test sets of the pure RandomForest classifier is: \t", test_acc, "%")
    print("The accuracy in train sets of the pure RandomForest classifier is: \t", train_acc, "%")
    test_acc = np.mat(test_acc)
    train_acc = np.mat(train_acc)

# Classifier
    clf = RandomForestClassifier(n_estimators=tree_number)

# train model / ravel to flatten the array structure from [[]] to []
    target = train_data[:,0].ravel()
    train = train_data[:,1:]
    model = clf.fit(train, target)

# modify the test_data, so the number of attributes match with the training data (missing label column)

# evaluate on testdata
    output = model.predict(test_data)

    # pd.DataFrame({"ImageId": range(1, len(output)+1), "Label": output}).to_csv('out.csv', index=False, header=True)
    return train_acc, test_acc


def chose_best_number(train, train_data, test, test_data, boundary=(50, 201)):
    number = list(range(boundary[0], boundary[1]))
    train_out = []
    test_out = []
    item = 0
    for num in number:
        train_out[item], test_out[item] = buildRandomForest(num, train, train_data, test, test_data)
        item += 1
    return train_out, test_out

# This part should add a function which could produce a picture of comparation
def maxacc(accuracy, best, item, bestitem):
    if accuracy > best:
        best = accuracy
        bestitem = item
    return best, bestitem


if __name__ == '__main__':
    train, train_data, test, test_data = input_and_visual('train.csv', 'test.csv')
    train_answer, test_answer = chose_best_number(train, train_data, test, test_data)
    best_train = 0
    best_test = 0
    for i in range(len(test_answer)[0]):
        item = i
        print("item:{} accuracy in train setting is:{} accuracy in test setting is: {}". format(item, answer[item, 0], answer[item, 1]))
        #best_train, best_train_item = maxacc(answer[item, 0], best_train, item, best_train_item)
        best_test, best_test_item = maxacc(test_answer[item, 1], best_test, item, best_test_item)
    print("the max accuracy : {} in item = {}".format(best_test, best_test_item))