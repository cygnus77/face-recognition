import os
import numpy as np
import glob
import cv2
import re

for infile in glob.glob("./*.jpg"):
    img = cv2.imread(infile)
    h, w, d = img.shape
    m = max(h,w)
    scale = 1.
    if m > 600:
        scale = 600./m
    outfile = re.sub("\\.jpg", ".png", infile)

    print("{}->{}: {}x{} => {}".format(infile,outfile, w, h, scale))

    img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(outfile, img)