#!/usr/bin/python3

from PIL import Image, ImageDraw
from datetime import datetime


def get_rgb_from_bw_percentage(bw_percentage, ratios={}):
    bw_value = float(bw_percentage * 255)  # higher (darker color) on lower (less happy) ratio (0 is black, 255 is white)
    rgb = [bw_value, bw_value, bw_value]
    if ratios.get('R'):
        rgb[0] = ratios['R'] * bw_value
        rgb[1] = ratios['G'] * bw_value
        rgb[2] = ratios['B'] * bw_value
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _get_timestamp():
    return str(datetime.now().strftime("%Y%m%d_%H:%M:%S"))


class TwitterImage:
    def __init__(self, x_dim, y_dim, filename, ext='png'):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.current_x = 0
        self.current_y = 0
        self.filename = filename
        self.ext = ext
        self.save_path = f"{filename}_{_get_timestamp()}.{ext}"
        self.image = Image.new(mode="RGB", size=(x_dim, y_dim))
        self.pixels = self.image.load()

    def _check_if_need_reset(self):
        if self.current_y >= self.y_dim:
            return self.get_new_image()
        return None

    def color_next_pixel(self, rgb, pixel_size=1):
        for i in range(0, pixel_size):
            for j in range(0, pixel_size):
                self.pixels[self.current_x + i, self.current_y + j] = rgb
        self.image.save(self.save_path)
        if self.current_x + pixel_size < self.x_dim:  # ex. 0-4 less than 5
            self.current_x = self.current_x + pixel_size
        else:
            self.current_x = 0
            self.current_y += 1

    def get_new_image(self):
        return TwitterImage(self.x_dim, self.y_dim, self.filename, ext=self.ext)
