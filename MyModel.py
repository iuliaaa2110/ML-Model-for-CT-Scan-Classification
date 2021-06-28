from copy import deepcopy
import numpy as np
from keras.constraints import max_norm
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, ELU
from keras.models import Sequential
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau


def read_image(path):
    """
    Retin imaginea sub forma de array.
    O impart la 255 deoarece sunt 256 de pixeli (de la 0 la 255) si o sa le asociez numai valori intre 0 si 1.
    :param path:
    :return:
    """
    img_mat = image.load_img(path, target_size=(50, 50, 1), grayscale=True)
    img_mat = image.img_to_array(img_mat)
    img_mat = img_mat / 255
    return img_mat


def read_data(file_name):
    """
    Functia va fi apelata pentru train si validation data
    :param file_name: fisierul .txt in care gasesc inputul
    :return: images (retinute intr un np.array) si labels
    """
    input_file = open(f'kaggle_data/{file_name}.txt')
    images, labels = [], []
    line = input_file.readline()

    while line != '':
        image_name, label = line.split(',')
        labels.append(int(label))

        path = f'kaggle_data/{file_name}/' + image_name
        images.append(read_image(path))

        line = input_file.readline()

    return np.array(images), labels


def read_test(file_name):
    """
    functia va fi apelata doar pt test data, aici nu mai citesc labels si am in plus lista image_names
    care sa imi retina numele imaginilor pt a putea face zip cu rezultatele inainte sa le afisez la final.
    :param file_name:
    :return: lista cu numele imaginilor si un np.array cu images
    """
    input_file = open(f'kaggle_data/{file_name}.txt')
    image_names = []
    images = []
    line = input_file.readline()

    while line != '':
        image_name = line.split()[0]
        image_names.append(image_name)
        path = f'kaggle_data/{file_name}/' + image_name
        images.append(read_image(path))

        line = input_file.readline()

    return image_names, np.array(images)


def cnn(train_images, images_to_predict, valid_images, train_labels, valid_labels, monitor='loss'):
    """
    :param train_images:
    :param images_to_predict:
    :param valid_images:
    :param train_labels:
    :param valid_labels:
    :param monitor: va fi 'val_loss' atunci cand avem validation data si loss altfel.
    :return:
    """
    print('create model')
    model = Sequential()
    # 1
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(50, 50, 1), kernel_initializer='he_normal',
                     kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(ELU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 2
    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(50, 50, 1), kernel_initializer='he_normal',
                     kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(ELU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 3
    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(50, 50, 1), kernel_initializer='he_normal',
                     kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(ELU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 4
    model.add(
        Conv2D(256, kernel_size=(3, 3), input_shape=(50, 50, 1), kernel_initializer='he_normal', activation='relu',
               kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 5
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005),
                    kernel_constraint=max_norm(3),
                    bias_constraint=max_norm(3)))
    # 6
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001),
                    kernel_constraint=max_norm(3),
                    bias_constraint=max_norm(3)))
    model.add(ELU(alpha=0.1))
    # 7
    model.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))

    print('compile')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('train')
    nparray_train_labels = np.array(train_labels)
    nparray_valid_labels = np.array(valid_labels)

    rlrop = ReduceLROnPlateau(monitor=monitor, mode='auto', patience=2, verbose=1, factor=0.2)

    if not valid_labels:
        model.fit(train_images, nparray_train_labels, epochs=15, callbacks=[rlrop])
    else:
        model.fit(train_images, nparray_train_labels, epochs=15,
                  validation_data=(valid_images, nparray_valid_labels), callbacks=[rlrop])

    print('predict')
    return model.predict_classes(images_to_predict)


def write_output(fileName, image_names, predicted_labels):
    g = open(fileName, 'w')
    g.write('id,label\n')
    for img, label in zip(image_names, predicted_labels):
        g.write(img + ',' + str(label) + '\n')


def run():
    print('read')

    train_images, train_labels = read_data('train')
    valid_images, valid_labels = read_data('validation')
    test_image_names, test_images = read_test('test')

    predicted_labels = cnn(train_images, test_images, valid_images, train_labels, valid_labels, 'val_loss')

    write_output('submission.txt', test_image_names, predicted_labels)


def run_concatenated():
    """
    Concatenez train data cu validation data si antrenez modelul pe intreaga multime.
    :return:
    """
    print('read')

    train_images, train_labels = read_data('train')
    valid_images, valid_labels = read_data('validation')
    test_image_names, test_images = read_test('test')

    train_labels = train_labels + deepcopy(valid_labels)
    train_images = np.concatenate([train_images, deepcopy(valid_images)])

    predicted_labels = cnn(train_images, test_images, [], train_labels, [])

    write_output('submission.txt', test_image_names, predicted_labels)


if __name__ == '__main__':
    # run()
    run_concatenated()

