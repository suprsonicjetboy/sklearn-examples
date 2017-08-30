import os
import numpy as np
import pandas as pd
import time
from sklearn import ensemble, externals

def main():
    START_TIME = time.time()

    SPECIES = ['setosa', 'versicolor', 'virginica']
    FEATURES = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ]
    LABEL = 'species'

    TRAIN_FILE = "./dataset/iris_training.csv"
    TEST_FILE = "./dataset/iris_test.csv"

    OUTPUT_DIR = "./models"
    OUTPUT_FILE = "{0}/iris-model.ckpt".format(OUTPUT_DIR)

    # Dataset
    training_dataset = pd.read_csv(TRAIN_FILE)
    test_dataset = pd.read_csv(TEST_FILE)

    # Shuffle test dataset
    #test_dataset= test_dataset.sample(frac=1).reset_index(drop=True)

    # Training data
    train_x = np.array(training_dataset[FEATURES].astype(np.float32))
    train_y = np.array(training_dataset[LABEL].astype(np.float32))

    # Test data
    test_x = np.array(test_dataset[FEATURES].astype(np.float32))
    test_y = np.array(test_dataset[LABEL].astype(np.float32))


    print("\n---------- INFORMATION ----------")
    print("FEATURES       : {0}".format(FEATURES))
    print("LABEL          : {0}".format(LABEL))
    print("TRAINING FILE  : {0}".format(TRAIN_FILE))
    print("TRAINING DATA  : {0}".format(len(training_dataset)))
    print("TEST FILE      : {0}".format(TEST_FILE))
    print("TEST DATA      : {0}".format(len(test_dataset)))
    print("OUTPUT         : {0}".format(OUTPUT_FILE))


    print("\n------------ ALGORITHM ------------")
    print("Randam Forest")
    model = ensemble.RandomForestClassifier()

    # Reconstruct the model 
    # model = externals.joblib.load(OUTPUT_FILE)

    model.fit(train_x, train_y)
    importances = model.feature_importances_


    print("\n------------ IMPORTANCES ------------")
    for i in range(len(FEATURES)):
        print("{0}: {1} ".format(FEATURES[i].ljust(15), importances[i]))


    result = model.predict_proba(test_x)
    print("\n------------ PREDICTION 1 ------------")
    print(result[:10])


    result = model.predict(test_x)
    print("\n------------ PREDICTION 2 ------------")
    print(result[:10])


    print("\n------------ PREDICTION 3 ------------")
    for i in range(10):
        if int(result[i]) == int(test_y[i]):
            print("True  => {1}   {0}  @@ {2}".format(int(result[i]), int(test_y[i]), SPECIES[int(test_y[i])]))
        else:
            print("False => {1}   {0}  @@ {2}".format(int(result[i]), int(test_y[i]), SPECIES[int(test_y[i])]))


    print("\n------------ SCORE ------------")
    print(model.score(test_x, test_y))


    # Persist the model 
    joblib = externals.joblib
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    joblib.dump(model, "{0}".format(OUTPUT_FILE))


    print("\n--------------------------------")
    print("Time: {0} sec".format(round(time.time() - START_TIME, 3)))
    print("--------------------------------")


if __name__ == '__main__':
    main()