import numpy as np
from PIL import Image
from sklearn import preprocessing

image_names = []


def get_images_and_labels(file_name = 'train'):
    input_file = open(f'kaggle_data/{file_name}.txt')
    # images_folder = Path(f'kaggle_data/{file_name}/')

    images, labels = [], []
    line = input_file.readline()
    global image_names
    image_names = []

    while line != '':
        # image_path = images_folder.glob(f"{image_name}.npy")
        if file_name == 'test':
            image_name = line.split()[0]
            label = '0'
        else:
            image_name, label = line.split(',')
        im = Image.open(f'kaggle_data/{file_name}/' + image_name).convert("RGB")
        im = np.array(im, dtype='uint8')

        # io.imshow(im.astype(np.uint8))
        # io.show()

        image_names.append(image_name)
        images.append(im)
        labels.append(int(label))
        line = input_file.readline()

    # transform într-un array
    images = np.array(images)
    return images, labels


def load_files():
    return get_images_and_labels('train') + get_images_and_labels('validation')


def normalize_data(train_data, test_data, norm_type=None):
    if norm_type is None:
        return train_data, test_data

    if norm_type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        return scaler.transform(train_data), scaler.transform(test_data)

    if norm_type == 'min_max':
        return (preprocessing.minmax_scale(train_data, axis=-1),
                preprocessing.minmax_scale(test_data, axis=-1))

    if norm_type == 'l1':
        return (preprocessing.normalize(train_data, norm='l1'),
                preprocessing.normalize(test_data, norm='l1'))

    if norm_type == 'l2':
        return (preprocessing.normalize(train_data, norm='l2'),
                preprocessing.normalize(test_data, norm='l2'))


def reshape(images):
    nsamples, nx, ny, nz = images.shape
    images = images.reshape((nsamples, nx * ny * nz))
    return images


def write_output(valid_preds):
    # output
    g = open('submission.txt', 'w')
    g.write('id,label\n')
    for img, label in zip(image_names, valid_preds):
        g.write(img + ',' + str(label) + '\n')