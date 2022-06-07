import os
import os.path
import glob
import natsort

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.webp'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, log, split_file, testing_set):
    left_fold = 'img_webp/'
    right_fold = 'img_stereo_webp/'
    disp_L = 'disp_occ_0_exr/'


    # left_train = [f for f in glob.glob(os.path.join(filepath, 'train', left_fold) + "*.webp")]
    # left_train = natsort.natsorted(left_train)
    # # print(left_train[:5])
    #
    # right_train = [f for f in glob.glob(os.path.join(filepath, 'train', right_fold) + "*.webp")]
    # right_train = natsort.natsorted(right_train)
    # # print(right_train[:5])
    #
    # left_train_disp = [f for f in glob.glob(os.path.join(filepath, 'train', disp_L) + "*.exr")]
    # left_train_disp = natsort.natsorted(left_train_disp)
    # # print(left_train_disp[:5])



    train = [x for x in os.listdir(os.path.join(filepath, 'train', left_fold)) if is_image_file(x)]
    train = natsort.natsorted(train)
    left_train = [os.path.join(filepath, 'train', left_fold, img) for img in train]
    right_train = [os.path.join(filepath, 'train', right_fold, img[-16:-5] + '_stereo' + img[-5:]) for img in train]
    left_train_disp = [os.path.join(filepath, 'train', disp_L, img[-16:-9] + '_disp.exr') for img in train]

    # left_val = [f for f in glob.glob(os.path.join(filepath, 'valid', left_fold) + "*.webp")]
    # left_val = natsort.natsorted(left_val)
    # # print(left_val[:5])
    #
    # right_val = [f for f in glob.glob(os.path.join(filepath, 'valid', right_fold) + "*.webp")]
    # right_val = natsort.natsorted(right_val)
    # # print(right_val[:5])
    #
    # left_val_disp = [f for f in glob.glob(os.path.join(filepath, 'valid', disp_L) + "*.exr")]
    # left_val_disp = natsort.natsorted(left_val_disp)
    # # print(left_val_disp[:5])


    val = [x for x in os.listdir(os.path.join(filepath, testing_set, left_fold)) if is_image_file(x)]
    val = natsort.natsorted(val)
    left_val = [os.path.join(filepath, testing_set, left_fold, img) for img in val]
    right_val = [os.path.join(filepath, testing_set, right_fold, img[-16:-5] + '_stereo' + img[-5:]) for img in val]
    left_val_disp = [os.path.join(filepath, testing_set, disp_L, img[-16:-9] + '_disp.exr') for img in val]

    return left_train, right_train, left_train_disp, left_val, right_val, left_val_disp
