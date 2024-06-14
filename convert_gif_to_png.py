import multiprocessing
from glob import glob

from PIL import Image


def gif_to_png(label):
    with Image.open(label) as image:
        rgb_image = image.convert("RGB")
        rgb_image.save(label.replace("gif", "jpg"), format="JPEG")


if __name__ == "__main__":
    labels = glob("dataset/labels/*.gif")
    with multiprocessing.Pool() as pool:
        pool.map(gif_to_png, labels)
