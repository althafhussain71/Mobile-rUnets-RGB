import os
from PIL import Image, ImageFilter

path = "dataset\\flood\\train\\"
copy_to_path = "dataset\\flood\\train_smooth_masks\\"

for filename in os.listdir(path):
    img = Image.open(os.path.join(path, filename)) # images are color images
    # applying filter and saving to the directory copy_to_path
    img = img.filter(ImageFilter.SMOOTH)
    img.save(copy_to_path+filename)