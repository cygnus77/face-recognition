import os
import io
import glob
from lxml import etree
import tensorflow as tf
from object_detection.utils import dataset_util
import numpy as np
import cv2


class_id_map = {'aditya': 1, 'rachana': 2, 'anand': 3, 'sanju': 4, 'ziggy': 5 }

def processFolder(dir):
    writer = tf.python_io.TFRecordWriter("{}_instances.record".format(dir))
    for f in glob.glob(os.path.join(dir, '*.xml')):
        with open(f) as fd:
            t = etree.parse(fd)
        #print(etree.tostring(t, pretty_print=True).decode("utf-8"))

        image_filename = t.xpath('/annotation/filename/text()')[0]
        
        image_path = os.path.join(os.path.dirname(f), image_filename)
        with open(image_path, 'rb') as imfd:
            img_bytes = io.BytesIO(imfd.read())

        img = cv2.imread(image_path)
        h, w, d = img.shape

        classes_text = []
        classes = []
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []

        objects = t.xpath('/annotation//object')
        for obj in objects:
            class_text = obj.xpath('name/text()')[0]
            classes_text.append(class_text.encode('utf-8'))
            classes.append(class_id_map[class_text])
            xmins.append(float(obj.xpath('bndbox/xmin/text()')[0]) / w)
            ymins.append(float(obj.xpath('bndbox/ymin/text()')[0]) / h)
            xmaxs.append(float(obj.xpath('bndbox/xmax/text()')[0]) / w)
            ymaxs.append(float(obj.xpath('bndbox/ymax/text()')[0]) / h)
        

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(h),
            'image/width': dataset_util.int64_feature(w),
            'image/filename': dataset_util.bytes_feature(image_filename.encode('utf-8')),
            'image/source_id': dataset_util.bytes_feature(image_filename.encode('utf-8')),
            'image/encoded': dataset_util.bytes_feature(img_bytes.getvalue()),
            'image/format': dataset_util.bytes_feature(b'png'),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
        writer.write(tf_example.SerializeToString())

    writer.close()

def main(_):
    processFolder('./val')
    processFolder('./train')

if __name__ == '__main__':
    tf.app.run()

#processFolder('./train')
