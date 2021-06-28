from pathlib import Path
from matplotlib import path, image
from sklearn import preprocessing
import numpy as np
from PIL import Image
import os, sys
import cv2
from skimage import io
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import main_utils


def SVM1(train_images, train_labels, valid_images, valid_labels):
    # 38, 9333333333
    train_images, valid_images = main_utils.normalize_data(train_images, valid_images, 'l2')

    # SVM:

    # pas1
    clf = SVC(C=1.0, kernel='linear') # 38, 9333333333

    # pas2
    print("incepe trainingul")
    clf.fit(train_images, train_labels)

    # pas3
    print('incep predictiile')
    valid_preds = clf.predict(valid_images)

    # output
    g = open('valid_preds.out', 'w')
    for label in valid_preds:
        g.write(str(label) + '\n')

    # clf.score(valid_pred, valid_labels)

    accuracy = accuracy_score(valid_preds, valid_labels)
    print("Accuracy:", accuracy * 100)


def SVM2(train_images, train_labels, valid_images, valid_labels):
    # 43.53333333333333
    # normalize:
    print("incepe normalizarea")
    train_images, valid_images = main_utils.normalize_data(train_images, valid_images, 'l2')

    # SVM:

    # pas1
    print('Creez SVC ul')
    clf = SVC(C=1.0, kernel='rbf')     # 43.53333333333333

    # pas2
    print("incepe trainingul")
    clf.fit(train_images, train_labels)

    # pas3
    print('incep predictiile')
    valid_preds = clf.predict(valid_images)

    # accuracy
    accuracy = accuracy_score(valid_preds, valid_labels)
    print("Accuracy:", accuracy*100)

    # output
    g = open('valid_preds2.out', 'w')
    for label in valid_preds:
        g.write(str(label) + '\n')

    # clf.score(valid_pred, valid_labels)


def SVM3(train_images, train_labels, valid_images, valid_labels):
    # 43.37777777777777
    # normalize:
    print("incepe normalizarea")
    # train_images, valid_images = normalize_data(train_images, valid_images, 'l1')  # 34.57777777777778, cu C = 2
    # cu gamma = 0,001
    train_images, valid_images = main_utils.normalize_data(train_images, valid_images, 'l2')  # 35.46666666666667 cu C = 2,
    # gamma = 0,001

    # SVM:

    # pas1
    print('Creez SVC ul')
    clf = SVC(C=1.0, gamma=10, kernel='rbf')

    # pas2
    print("incepe trainingul")
    clf.fit(train_images, train_labels)

    # pas3
    print('incep predictiile')
    valid_preds = clf.predict(valid_images)

    # accuracy
    accuracy = accuracy_score(valid_preds, valid_labels)
    print("Accuracy:", accuracy*100)

    # output
    g = open('valid_preds3.out', 'w')
    for label in valid_preds:
        g.write(str(label) + '\n')

    # clf.score(valid_pred, valid_labels)


def SVM4(train_images, train_labels, valid_images):
    # 43.77777777777778
    # normalize:
    print("incepe normalizarea")
    train_images, valid_images = main_utils.normalize_data(train_images, valid_images, 'l2')
    # cu min_max da maai putin:  43.044444444444444

    # SVM:

    # pas1
    print('Creez SVC ul')
    clf = SVC(C=1.0, gamma=7, kernel='poly')
    # pas2
    print("incepe trainingul")
    clf.fit(train_images, train_labels)

    # pas3
    print('incep predictiile')
    valid_preds = clf.predict(valid_images)

    # accuracy
    # accuracy = accuracy_score(valid_preds, valid_labels)
    # print("Accuracy:", accuracy*100)

    # output
    main_utils.write_output(valid_preds)
    # g = open('valid_preds4.out', 'w')
    # for label in valid_preds:
    #     g.write(str(label) + '\n')


def quadratic_discriminant_analysis(train_images, train_labels, valid_images, valid_labels):
    # 38.2
    # normalize:
    print("incepe normalizarea")
    train_images, valid_images = main_utils.normalize_data(train_images, valid_images, 'l2')

    # pas1
    print('Creez Quadratic Discriminant Analysis ul')
    clf = QuadraticDiscriminantAnalysis()

    # pas2
    print("incepe trainingul")
    clf.fit(train_images, train_labels)

    # pas3
    print('incep predictiile')
    valid_preds = clf.predict(valid_images)

    # accuracy
    accuracy = accuracy_score(valid_preds, valid_labels)
    print("Accuracy:", accuracy*100)

    # output
    g = open('quadratic_preds.out', 'w')
    for label in valid_preds:
        g.write(str(label) + '\n')


def linear_discriminant_analysis(train_images, train_labels, valid_images, valid_labels):
    # normalize:
    print("incepe normalizarea")
    train_images, valid_images = main_utils.normalize_data(train_images, valid_images, 'l2')

    # pas1
    print('Creez Quadratic Discriminant Analysis ul')
    clf = LinearDiscriminantAnalysis(solver='lsqr', n_components=3, shrinkage=float)

    # pas2
    print("incepe trainingul")
    clf.fit(train_images, train_labels)

    # pas3
    print('incep predictiile')
    valid_preds = clf.predict(valid_images)

    # accuracy
    accuracy = accuracy_score(valid_preds, valid_labels)
    print("Accuracy:", accuracy*100)

    # output
    g = open('quadratic_preds.out', 'w')
    for label in valid_preds:
        g.write(str(label) + '\n')


if __name__ == '__main__':
    train_images, train_labels, valid_images, _ = main_utils.load_files()

    # reshape:
    nsamples, nx, ny, nz = train_images.shape
    train_images = train_images.reshape((nsamples, nx * ny * nz))

    nsamples, nx, ny, nz = valid_images.shape
    valid_images = valid_images.reshape((nsamples, nx * ny * nz))

    # SVM1(train_images, train_labels, valid_images, valid_labels) # 38.9333333333
    # SVM2(train_images, train_labels, valid_images, valid_labels) # 43.53333333333333
    # SVM3(train_images, train_labels, valid_images, valid_labels) # 43.37777777777777
    SVM4(train_images, train_labels, valid_images)  # 43.77777777777778
    # normalizare l2 cu gama 7:  43.77777777777778
    # normalizare standard + gama 6:  40.666666666666664

    # quadratic_discriminant_analysis(train_images, train_labels, valid_images, valid_labels) # 38.2