import multiprocessing
from glob import glob

from PIL import Image

labels = glob("dataset/labels/*.gif")


def gif_to_png(label):
    with Image.open(label) as image:
        rgb_image = image.convert("RGB")
        rgb_image.save(label.replace("gif", "jpg"), format="JPEG")


with multiprocessing.Pool() as pool:
    pool.map(gif_to_png, labels)

print("Done UwU!!!!")
