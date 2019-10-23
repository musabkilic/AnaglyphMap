import tqdm
import argparse
import numpy as np
from PIL import Image

class AnaglyphMap:
    def __init__(self, dem_file, output_name="output.png", observer_alt=4000, map_plane_alt=0.0, eye_spacing=750, nadir=0.5, keep_all=True):
        self.map = Image.open(dem_file)
        self.w, self.h = self.map.size
        self.output_name = output_name
        self.observer_alt = observer_alt
        self.map_plane_alt = map_plane_alt
        self.eye_spacing = eye_spacing
        self.nadir = nadir
        self.keep_all = keep_all

    @staticmethod
    def color_translation(elevation, minBD, maxBD, colorA, colorB):
        for i in range(3):
            yield int(colorA[i]+(colorB[i] - colorA[i])*(elevation-minBD)/(maxBD-minBD))

    def colorize(self):
        self.col_im = Image.new("RGB", (self.w, self.h))
        self.min_d = +float("inf")
        self.max_d = -float("inf")
        for y in tqdm.tqdm(range(self.h), desc="Colorizing Image"):
            for x in range(self.w):
                elevation = self.map.getpixel((x, y))
                self.min_d = min(self.min_d, elevation)
                self.max_d = max(self.max_d, elevation)

                if elevation <= 0:
                    self.col_im.putpixel((x, y), (21, 172, 191))
                elif 0 < elevation <= 200:
                    self.col_im.putpixel((x, y), (81, 201, 38))
                elif 200 < elevation <= 500:
                    self.col_im.putpixel((x, y), tuple(list(self.color_translation(elevation, 200, 500, (81, 201, 38), (160, 242, 130)))))
                elif 500 < elevation <= 1000:
                    self.col_im.putpixel((x, y), tuple(list(self.color_translation(elevation, 500, 1000, (160, 242, 130), (237, 243, 100)))))
                elif 1000 < elevation <= 1500:
                    self.col_im.putpixel((x, y), tuple(list(self.color_translation(elevation, 1000, 1500, (237, 243, 100), (240, 173, 25)))))
                elif 1500 < elevation <= 2000:
                    self.col_im.putpixel((x, y), tuple(list(self.color_translation(elevation, 1500, 2000, (240, 173, 25), (203, 179, 130)))))
                elif 2000 < elevation <= 2500:
                    self.col_im.putpixel((x, y), tuple(list(self.color_translation(elevation, 2000, 2500, (203, 179, 130), (149, 119, 35)))))
                else:
                    self.col_im.putpixel((x, y), (255, 255, 255))

    def split2LNR(self):
        self.left_im = Image.new("RGB", (self.w, self.h))
        self.right_im = Image.new("RGB", (self.w, self.h))

        for y in tqdm.tqdm(range(self.h), desc="Splitting Image Into Left and Right"):
            for x in range(self.w):
                pix = self.col_im.getpixel((x, y))
                elevation = self.map.getpixel((x, y))
                if elevation <= 0:
                    self.left_im.putpixel((x, y), pix)
                    self.right_im.putpixel((x, y), pix)
                else:
                    left_new_x  = min(self.w-1, max(0, x-int(self.nadir * self.eye_spacing * (elevation - self.map_plane_alt) / (self.observer_alt - elevation))))
                    right_new_x = min(self.w-1, max(0, x+int((1 - self.nadir) * self.eye_spacing * (elevation - self.map_plane_alt) / (self.observer_alt - elevation))))
                    self.left_im.putpixel((left_new_x, y), pix)
                    self.right_im.putpixel((right_new_x, y), pix)

    def lnr23D(self):
        A = np.matrix([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        B = np.matrix([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        result = (np.tensordot(np.array(self.left_im)  / 255, A, axes=([2],[0])) + \
               np.tensordot(np.array(self.right_im) / 255, B, axes=([2],[0]))) * 255
        return Image.fromarray(result.astype(np.uint8))

    def process(self):
        self.colorize()
        self.split2LNR()
        self.lnr23D().save(self.output_name)

        if self.keep_all:
            self.col_im.save("col_im.png")
            self.left_im.save("left_im.png")
            self.right_im.save("right_im.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn DEM(Digital Elevation Model) Files to 3D Anaglyph Images")
    parser.add_argument("--dem-file", type=str, required=True, help="DEM file to process")
    parser.add_argument("--output", "-o", default="output.png", type=str, help="Output file name(default: output.png)")
    parser.add_argument("--observer-alt", "-a", default=4000, type=int, help="Observer Altitude(default: 4000)")
    parser.add_argument("--map-plane-alt", "-p", default=0.0, type=float, help="Map Plane Altitude(default: 0.0)")
    parser.add_argument("--eye-spacing", "-s", default=750, type=int, help="Spacing Between Left and Right Eyes(default: 750)")
    parser.add_argument("--nadir", "-n", default=0.5, type=float, help="Ratio Between Left and Right Eyes(default: 0.50)")
    parser.add_argument("--keep-all", "-k", default=True, action="store_true", help="Save All Processed Images(default: True)")
    args = parser.parse_args()
    AnaglyphMap = AnaglyphMap(args.dem_file, output_name=args.output, observer_alt=args.observer_alt, \
                              map_plane_alt=args.map_plane_alt, eye_spacing=args.eye_spacing, nadir=args.nadir, \
                              keep_all=args.keep_all)
    AnaglyphMap.process()
