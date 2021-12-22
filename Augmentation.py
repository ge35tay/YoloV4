# first part

# MainCode
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from matplotlib import image
import cv2
from PIL import Image
import PIL
import array


def augmentation(image_file, text_file, destination, n):
    image0 = PIL.Image.open(image_file)
    x_image, y_image = image0.size
    image1 = image.imread(image_file)
    lines = open(text_file).readlines()
    image2 = np.expand_dims(image1, axis=0)
    boxes = []
    class_id = []
    with open(text_file) as fuck:

        for line in fuck:
            # line = line.split()
            box = np.array(list(line.split(';')))  # map the position as float
            x1 = (2 * float(box[1]) - float(box[3])) * 0.5 * x_image
            x2 = (2 * float(box[1]) + float(box[3])) * 0.5 * x_image
            y1 = (2 * float(box[2]) - float(box[4])) * 0.5 * y_image
            y2 = (2 * float(box[2]) + float(box[4])) * 0.5 * y_image
            boxes += [ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)]
            """
            BoundingBox x1,y1, x2, y2, x_center, y_center in the dataset devide by the length and height of image
            """
            class_id.append(box[0])
    # seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.05*255), iaa.Affine(translate_px={"x": (1, 5)}),iaa.Invert(0.05, per_channel=True) ])
    # seq = iaa.Sequential([iaa.Invert(0.5, per_channel=True),iaa.Affine(translate_px={"x": (1, 5)}),iaa.Invert(0.05, per_channel=True)])

    skip = lambda aug: iaa.Sometimes(1, aug)
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = skip(iaa.Sequential([
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images

        # crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.1))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale images to a value of 80 to 120% of their original size
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode='constant'
        )),

        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 5),
                   [
                       # Convert some images into their superpixel representation,
                       # sample between 20 and 200 superpixels per image, but do
                       # not replace all superpixels with their average, only
                       # some of them (p_replace).

                       # superpixels  --- For the raw image, apply a superpixel generation method to obtain superpixel
                       # cells, for each of cells, computer the average pixel values for all pixels in cell and assign the
                       # computed average value to all pixels in C

                       sometimes(
                           iaa.Superpixels(
                               p_replace=(0, 1.0),
                               n_segments=(20, 200))
                       ),

                       # Blur each image with varying strength using
                       # gaussian blur (sigma between 0 and 3.0),
                       # average/uniform blur (kernel size between 2x2 and 7x7)
                       # median blur (kernel size between 3x3 and 11x11).
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),
                           iaa.AverageBlur(k=(2, 7)),
                           iaa.MedianBlur(k=(3, 11)),
                       ]),

                       # Sharpen each image, overlay the result with the original
                       # image using an alpha between 0 (no sharpening) and 1
                       # (full sharpening effect). make the shape boundary clear
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                       # Same as sharpen, but for an embossing effect.
                       # Image embossing is a computer graphics technique in which each pixel of an image is replaced either
                       # by a highlight or a shadow, depending on light/dark boundaries on the original image.
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                       # Search in some images either for all edges or for
                       # directed edges. These edges are then marked in a black
                       # and white image and overlayed with the original image
                       # using an alpha of 0 to 0.7.
                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0, 0.7)),
                           iaa.DirectedEdgeDetect(
                               alpha=(0, 0.7), direction=(0.0, 1.0)
                           ),
                       ])),

                       # Add gaussian noise to some images.
                       # In 50% of these cases, the noise is randomly sampled per
                       # channel and pixel.
                       # In the other 50% of all cases it is sampled once per
                       # pixel (i.e. brightness change).
                       iaa.AdditiveGaussianNoise(
                           loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                       ),

                       # Either drop randomly 1 to 10% of all pixels (i.e. set
                       # them to black) or drop them on an image with 2-5% percent
                       # of the original size, leading to large dropped
                       # rectangles.
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.02, 0.05),
                               per_channel=0.2),  # drop a rectangles area as the pixel value 0
                       ]),

                       # Invert each image's channel with 5% probability.
                       # This sets each pixel value v to 255-v.
                       iaa.Invert(0.05, per_channel=True),  # invert color channels

                       # Add a value of -10 to 10 to each pixel.
                       iaa.Add((-10, 10), per_channel=0.5),

                       # Change brightness of images (50-150% of original value).
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),

                       # Improve or worsen the contrast of images.
                       iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),

                       # Convert each image to grayscale and then overlay the
                       # result with the original with random alpha. I.e. remove
                       # colors with varying strengths.
                       iaa.Grayscale(alpha=(0.0, 1.0)),

                       # In some images move pixels locally around (with random
                       # strengths alpha).
                       sometimes(
                           iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                       ),

                       # In some images distort local areas with varying strength.
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                   ],
                   # do all of the above augmentations in random order
                   random_order=True
                   )
    ],
        #  do all of the above augmentations in random order
        random_order=True
    ))

    images_aug, bbs_aug = seq(images=image2, bounding_boxes=boxes)
    images_aug = np.squeeze(images_aug, axis=0)
    PIL_image = Image.fromarray(images_aug.astype('uint8'), 'RGB')
    path_image = destination + "/" + image_file.split(".jpg")[0].split("/")[-1] + "aug" + n + ".jpg"   # after aug
    path_text = destination + "/" + image_file.split(".jpg")[0].split("/")[-1] + "aug" + n + ".txt"    # after aug
    #  path_text=image_file.split(".jpg")[0]+"aug"+n+".txt"
    PIL_image.save(path_image, "JPEG")
    f = open(path_text, "w")
    for i in range(len(bbs_aug)):
        a = (bbs_aug[i].x1 + bbs_aug[i].x2) / (2 * x_image)
        b = (bbs_aug[i].y1 + bbs_aug[i].y2) / (2 * y_image)
        c = (bbs_aug[i].x2 - bbs_aug[i].x1) / (x_image)
        d = (bbs_aug[i].y2 - bbs_aug[i].y1) / (y_image)
        round(12.3456, 2)
        annotation = str(class_id[i]) + " " + str(round(a, 8)) + " " + str(round(b, 8)) + " " + str(
            round(c, 8)) + " " + str(round(d, 8)) + "\n"
        f.write(annotation)
    f.close()


# second part
import os, random
import os.path
import shutil
from distutils.dir_util import copy_tree


def data_augmentation(folder, destination, n):
    if n >= 1:
        copy_tree(folder, destination)

    for j in range(int(n) - 1):
        l = os.listdir(folder)
        random.shuffle(l)
        for file in os.listdir(folder):
            if file.endswith(".jpg"):
                a = str(file)
                b = a.split(".")
                x = b[0]
                if os.path.isfile(folder + "/" + x + ".jpg"):

                    try:
                        print(1)
                        augmentation(folder + "/" + x + ".jpg", "D:/Leben_in_TUM/TUMPhoenix/GTSDB_jpg/gt.txt", destination, str(j + 1))
                    except:
                        continue

        # for file in os.listdir(folder):
        #    if file.endswith(".txt"):
        #        a = str(file)
        #        b = a.split(".")
        #        print(b)
        #        x = ""
        #        for f in range(len(b) - 1):
        #            x += str(b[f])
                # print(folder+x+".jpg")
        #        if os.path.isfile(folder + "/"  + ".ppm"):
        #            augmentation(folder + "/" + x + ".ppm", folder + "/" + ".txt", destination, str(j + 1))

    # n = n - int(n)
    # dirListing = os.listdir(folder)
    # dirListing = dirListing[6:]
    # m = len(dirListing)
    # l = os.listdir(folder)
    # random.shuffle(dirListing)
    # for k in range(int(m * n)):
    #    if dirListing[k].endswith(".txt"):
    #        a = str(dirListing[k])
    #        b = a.split(".")
    #        x = ""
    #        for f in range(len(b) - 1):
    #            x += str(b[f])
    #        print(k)
    #        print(x)
    #        print(folder + "" + '.ppm')
    #        if os.path.isfile(folder + "/" + ".ppm"):
    #            print(1)
    #            augmentation(folder + "/" + ".ppm", folder + "/" + ".txt", destination, "")


# folder = "/mnt/NewHDD/darknet-format/augmentation/loco_train"
folder = 'D:/Leben_in_TUM/TUMPhoenix/GTSDB_jpg'
# destination = "/mnt/NewHDD/darknet-format/augmentation/loco_train_augmented"
destination = 'D:/Leben_in_TUM/TUMPhoenix/GTSDB_Augmentation'
data_augmentation(folder, destination, 27.76)
