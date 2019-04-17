import numpy as np
import os
from PIL import Image
import pickle

def global_iteration(file_path, update=None):
    if not update:
        try:
            f = open(file_path, 'r', encoding='utf8')
            val = int(f.read())
            f.close()
            return val
        except:
            f = open(file_path, 'w', encoding='utf8')
            f.write(str(0))
            f.close()
            return 0
    else:
        f = open(file_path, 'w', encoding='utf8')
        f.write(str(update))
        f.close()

def load_image(dir, mode='RGB'):
    im = Image.open(dir).convert(mode).resize((105, 105), Image.LANCZOS)
    im = np.asarray(im)
    # if type is "L" (greyscale) apply the preprocessing phase
    if mode == 'L':
        # preprocessing image
        # 1 - image scaling (matrix with values in [0.0, 1.0])
        # 2 - image rescaling (matrix with values in [-1.0, 1.0])
        im = im / 255.0
        im = im - 0.5
        im = im * 2.0
        im = np.expand_dims(im, -1)
    return im

def load_dataset(tmp_dir, data_dir):
    if not os.path.exists(tmp_dir + '/train_set.pkl') or not os.path.exists(tmp_dir + '/test_set.pkl'):
        print('Loading dataset...')
        dataset = []
        category2index = {}
        index = 0
        for folder in os.listdir(data_dir):
            if folder not in category2index:
                category2index[folder] = index
                index += 1
            imgs = []
            directory = os.path.join(data_dir, folder)
            dir = os.path.join(directory, 'frontal')
            # print(folder)
            for img in os.listdir(dir):
                img = load_image(os.path.join(dir, img)) #RGB is the best choice
                imgs.append(img)
            dataset.append(imgs)
        # print(len(dataset))
        # print(category2index)
        train_set, test_set = split_dataset(dataset, category2index, train_size=0.75)
        del dataset
        pickle.dump(train_set, open(tmp_dir + '/train_set.pkl', 'wb'))
        pickle.dump(test_set, open(tmp_dir + '/test_set.pkl', 'wb'))
    else:
        print('Reloading train set and test set...')
        train_set = pickle.load(open(tmp_dir + '/train_set.pkl', 'rb'))
        test_set = pickle.load(open(tmp_dir + '/test_set.pkl', 'rb'))

    return train_set, test_set


def split_dataset(dataset, categories, train_size=0.75):
    train_set = []
    test_set = []
    train_categories = set(np.random.choice(list(categories.values()), size=round(len(categories) * train_size), replace=False))
    #print(train_categories)
    test_categories = set(categories.values()) - train_categories
    #print(test_categories)
    for category in train_categories:
        l = []
        for sample in dataset[category]:
            l.append(sample)
        train_set.append(l)
    for category in test_categories:
        l = []
        for sample in dataset[category]:
            l.append(sample)
        test_set.append(l)
    return train_set, test_set


def get_batch(train_set, batch_size):
    cat = np.random.choice(list(range(len(train_set))), size=batch_size, replace=False)
    #print(cat)
    label = np.zeros(batch_size)
    # If the inputs are from the same class, then the value of label is 1, otherwise label is 0
    label[:batch_size // 2] = 1
    batch = []
    for i in range(batch_size // 2):
        category = cat[i]
        random_index = np.random.randint(0, len(train_set[category]))
        img_1 = train_set[category][random_index]
        random_index = np.random.randint(0, len(train_set[category]))
        img_2 = train_set[category][random_index]
        batch.append((img_1, img_2))
    for i in range(batch_size // 2, batch_size):
        category_1 = cat[i]
        random_index = np.random.randint(0, len(train_set[category_1]))
        img_1 = train_set[category_1][random_index]
        category_2 = (category_1 + np.random.randint(1, len(train_set))) % len(train_set)
        img_2 = train_set[category_2][random_index]
        batch.append((img_1, img_2))
    return batch, label


def get_one_shot_test(test_set):
    n_classes = len(test_set)
    n_examples = len(test_set[0])
    cat = np.random.choice(list(range(n_classes)), size=n_classes, replace=False)
    random_indexes = np.random.randint(0, n_examples, size=n_examples)
    true_cat = cat[0]
    ex1, ex2 = np.random.choice(n_examples, replace=False, size=2)
    test = []
    label = np.zeros(n_classes)
    img_1 = test_set[true_cat][ex1]
    k = 0
    for random_index in random_indexes:
        if k == 0:
            img_2 = test_set[cat[k]][ex2]
        else:
            img_2 = test_set[cat[k]][random_index]
        test.append((img_1, img_2))
        k += 1
    label[0] = 1
    return test, label
