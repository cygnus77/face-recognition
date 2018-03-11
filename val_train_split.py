import os
import shutil
import glob
import re
import numpy as np

files = glob.glob('./*.png')
val = np.random.choice(files, size=len(files) // 5, replace = False)
train = [x for x in files if x not in val]

def move(lst, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    elif not os.path.isdir(dst_dir):
        return
    for f in lst:
        xml = re.sub('.png', '.xml', f)
        shutil.move(f, os.path.join(dst_dir, os.path.basename(f)))
        shutil.move(xml, os.path.join(dst_dir, os.path.basename(xml)))

move(val, './val')
move(train, './train')
