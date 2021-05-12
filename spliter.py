import os, shutil

orginal_path = "D:/dataset/ds images/train"
base_dir = "D:/dataset/ds images/kaggle"

train_cats_dir = base_dir + "/train/cats"
train_dogs_dir = base_dir + "/train/dogs"
test_cats_dir = base_dir + "/test/cats"
test_dogs_dir = base_dir + "/test/dogs"
validation_cats_dir = base_dir + "/validation/cats"
validation_dogs_dir = base_dir + "/validation/dogs"

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(orginal_path, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copy(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(orginal_path, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copy(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(orginal_path, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copy(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(orginal_path, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copy(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(orginal_path, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copy(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(orginal_path, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copy(src, dst)
