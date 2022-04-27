"""
Usage:
  # From tensorflow/object_detection_api/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

tf = tf.compat.v1

# flags = tf.app.flags
flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
flags.DEFINE_string('num_classes', '53', '28 or 53')
FLAGS = flags.FLAGS

electronic_devices = ['electronic device', 'smartphone', 'laptop', 'tablet', 'electric razor', 'battery charger',
                      'case headphones', 'electric cigarette', 'suspicious device', 'camera', 'electric toothbrush',
                      'curling iron', 'headphones', 'case with headphones', 'keyboard', 'acoustic system',
                      'big headphones']


def class_text_to_int_28(row_label):
    if row_label == 'flacon':
        return 1
    if row_label == 'tools':
        return 2
    if row_label == 'battery':
        return 3
    if row_label == 'cable':
        return 4
    if row_label == 'keys':
        return 5
    if row_label == 'pot':
        return 6
    if row_label == 'scissors':
        return 7
    if row_label == 'bottle':
        return 8
    if row_label == 'lighter':
        return 9
    if row_label == 'perfumery':
        return 10
    if row_label == 'gas cylinder':
        return 11
    if row_label == 'explosive':
        return 12
    if row_label == 'detonator':
        return 13
    if row_label == 'razor':
        return 14
    if row_label == 'watch':
        return 15
    if row_label == 'flashlight':
        return 16
    if row_label == 'ammunition':
        return 17
    if row_label == 'gun':
        return 18
    if row_label == 'grenade':
        return 19
    if row_label == 'screwdriver':
        return 20
    if row_label == 'jewelry':
        return 21
    if row_label == 'falshveer':
        return 22
    if row_label == 'mine':
        return 23
    if row_label == 'brass knuckles':
        return 24
    if row_label == 'knife':
        return 25
    if row_label == 'stun gun':
        return 26
    if row_label == 'hammer':
        return 27
    if row_label in electronic_devices:
        return 28
    else:
        return 0


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'smartphone':
        return 1
    if row_label == 'flacon':
        return 2
    if row_label == 'tablet':
        return 3
    if row_label == 'laptop':
        return 4
    if row_label == 'electric razor':
        return 5
    if row_label == 'tools':
        return 6
    if row_label == 'battery':
        return 7
    if row_label == 'battery charger':
        return 8
    if row_label == 'cable':
        return 9
    if row_label == 'metal item':
        return 10
    if row_label == 'coins':
        return 11
    if row_label == 'keys':
        return 12
    if row_label == 'footwear':
        return 13
    if row_label == 'pot':
        return 14
    if row_label == 'scissors':
        return 15
    if row_label == 'bottle':
        return 16
    if row_label == 'spoon':
        return 17
    if row_label == 'pen':
        return 18
    if row_label == 'lighter':
        return 19
    if row_label == 'perfumery':
        return 20
    if row_label == 'gas cylinder':
        return 21
    if row_label == 'acoustic system':
        return 22
    if row_label == 'explosive':
        return 23
    if row_label == 'umbrella':
        return 24
    if row_label == 'folding knife':
        return 25
    if row_label == 'glasses':
        return 26
    if row_label == 'detonator':
        return 27
    if row_label == 'fan':
        return 28
    if row_label == 'razor':
        return 29
    if row_label == 'watch':
        return 30
    if row_label == 'metal case':
        return 31
    if row_label == 'thermos':
        return 32
    if row_label == 'curling iron':
        return 33
    if row_label == 'flashlight':
        return 34
    if row_label == 'ammunition':
        return 35
    if row_label == 'gun':
        return 36
    if row_label == 'grenade':
        return 37
    if row_label == 'camera':
        return 38
    if row_label == 'screwdriver':
        return 39
    if row_label == 'headphones':
        return 40
    if row_label == 'book':
        return 41
    if row_label == 'binoculars':
        return 42
    if row_label == 'jewelry':
        return 43
    if row_label == 'falshveer':
        return 44
    if row_label == 'mine':
        return 45
    if row_label == 'brass knuckles':
        return 46
    if row_label == 'knife':
        return 47
    if row_label == 'stun gun':
        return 48
    if row_label == 'сable':
        return 9
    if row_label == 'hammer':
        return 49
    if row_label == 'electric device':
        return 50
    if row_label == 'tincan':
        return 51
    if row_label == 'plastic bottle':
        return 52
    if row_label == 'phone':
        return 53
    # else:
    #     return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    print(os.path.join(path, '{}'.format(group.filename)))
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        # print(fid)
        # print(fid.readline())
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

    filename = group.filename.encode('utf-8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # row = {'class': None}
    dict_functions = {
        '53': class_text_to_int,
        '28': class_text_to_int_28
    }
    for index, row in group.object.iterrows():
        # if row['class'] in electronic_devices:
        #     row['class'] = 'electronic device'
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf-8'))
        classes.append(dict_functions[FLAGS.num_classes](row['class']))

    print(filename)
    print(classes)
    print(classes_text)
    print(dataset_util.int64_list_feature(classes))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input, encoding='utf-8')
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    grouped = split(examples, 'filename')
    count = len(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
    print(f'Total objects - {count}')


if __name__ == '__main__':
    tf.app.run()
    print('Complete')
