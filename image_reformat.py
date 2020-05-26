"""

Author: Joey Yurgelon
"""

import os
import sys
from PIL import Image
from numpy import asarray


"""
This line imports SimpleImage for use here
This line depends on the Pillow package being installed
"""






def jpgs_in_dir(dir):
    """
    (provided, DO NOT MODIFY)
    Given the name of a directory, returns a list of the .jpg filenames
    within it.

    Input:
        dir (string): name of directory
    Returns:
        filenames(List[string]): names of jpg files in directory
    """
    filenames = []
    for filename in os.listdir(dir):
        if filename.endswith('.png'):
            filenames.append(os.path.join(dir, filename))
    return filenames


def load_images(dir):
    """
    (provided, DO NOT MODIFY)
    Given a directory name, reads all the .jpg files within it into memory and
    returns them in a list. Prints the filenames out as it goes.

    Input:
        dir (string): name of directory
    Returns:
        images (List[SimpleImages]): list of images in directory
    """
    images = []
    jpgs = jpgs_in_dir(dir)
    for filename in jpgs:
        print("Loading", filename)
        image = Image.open(filename)
        print("Format", image.format)
        print("Size", image.size)
        print("Mode", image.mode)
        images.append(image)
    return images


def main():
    # (provided, DO NOT MODIFY)
    args = sys.argv[1:]
    # We just take 1 argument, the folder containing all the images.
    # The load_images() capability is provided above.
    #images = load_images(args[0])

    images = []
    jpgs = jpgs_in_dir(args[0])
    print(jpgs)
    for filename in jpgs:
        print("Loading", filename)
        image = Image.open(filename)
        #print("Format", image.format)
        #print("Size", image.size)
        #print("Mode", image.mode)
        images.append(image)

    for i in range(len(images)):

        data = asarray(images[i])
        temp = data
        temp.flags.writeable = True
        #print(temp)
        temp[temp > 0] = 255
        dir = jpgs[i].split('/')
        #print(dir)
        im = Image.fromarray(temp)
        im.save('Corrected_Masks/' + dir[2])



if __name__ == '__main__':
    main()
