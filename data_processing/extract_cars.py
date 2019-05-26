"""
    - Script to extract cars from image
"""
from scipy import io as mat_io
from skimage import io as img_io

original_folder = 'datasets/training/original/'
extracted_folder = 'datasets/training/extracted/'
metas = 'datasets/cars_metas/cars_train_annos.mat'

if __name__ == '__main__':

    labels_meta = mat_io.loadmat(metas)

    for img_ in labels_meta['annotations'][0]:
        x_min = img_[0][0][0]
        y_min = img_[1][0][0]

        x_max = img_[2][0][0]
        y_max = img_[3][0][0]

        if len(img_) == 6:
            img_name = img_[5][0]
        elif len(img_) == 5:
            img_name = img_[4][0]
        try:
            img_in = img_io.imread(original_folder + img_name)
        except Exception:
            print("Error while reading!")
        else:
            # print(img_in.shape)
            cars_extracted = img_in[y_min:y_max, x_min:x_max]
            print(x_min, y_min, x_max, y_max, cars_extracted.shape, img_in.shape, img_name)

            img_io.imsave(extracted_folder + img_name, cars_extracted)

