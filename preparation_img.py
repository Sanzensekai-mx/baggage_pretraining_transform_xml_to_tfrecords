r"""
usage: preparation_img.py [-h] [-x XML_PATH] [-c OUTPUT_DIR] [-n NUM_CLASSES] [-i IMAGE_DIR]

Convert images and images xml labels into tfrecords format

optional arguments:
  -h, --help            show this help message and exit
  -x XML_PATH, --xml_path XML_PATH
                        Path to the folder where the image xml files stored.
                        If not specified, the CWD will be used.
  -c OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to dir there csv output and tfrecord file will appear
  -n NUM_CLASSES, --num_classes NUM_CLASSES
                        Num classes of items 53 or 28
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to directory with images according with their xml files

Example
python preparation_img.py -x C:\Users\SanZenSekai\PycharmProjects\baggage_training\baggage-object-detector\annotations\XML\first_try_1024\train -n 53
-i E:\Downloads\new_image_processing\3.baggage_1024x1024\pics -c E:\Downloads\new_image_processing\3.baggage_1024x1024
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys

import io
import collections
import xml_to_csv
import pandas as pd
import tensorflow as tf
from PIL import Image
from argparse import ArgumentParser
from object_detection.utils import dataset_util

tf = tf.compat.v1


def process_to_csv(image_xml_path: str, output_dir: str):
    """
    process xml to csv
    :param image_xml_path:
    :param output_dir:
    :return:
    """

    if not os.path.exists(image_xml_path):
        print('This path doesn\'t exist')
        sys.exit()
    if not os.path.isdir(image_xml_path):
        print('This path lead not to dir. Run with parameter "-h" to see help')
        sys.exit()
    # for directory in os.listdir(image_xml_path):
    #     dir_path = os.path.join(image_xml_path, directory)
    xml_df = xml_to_csv.xml_to_csv(image_xml_path)
    xml_dirname = os.path.basename(image_xml_path)
    num_of_same_csv_files = len([x for x in os.listdir(output_dir) if f'{xml_dirname}_objects_labels' in x])
    # print(os.listdir(os.path.split(output_dir)[0]))
    # print(num_of_same_csv_files)
    if num_of_same_csv_files == 0:
        out_csv_path = os.path.join(output_dir, f'{xml_dirname}_objects_labels.csv')
    else:
        out_csv_path = os.path.join(output_dir, f'{xml_dirname}_objects_labels{num_of_same_csv_files}.csv')
    # print(directory)
    if output_dir is not None and os.path.exists(output_dir) and os.path.isdir(output_dir):
        xml_df.to_csv(out_csv_path, index=None, encoding='utf-8')
    else:
        sys.exit()
    print(f'Successfully converted dir \"{xml_dirname}\" with xml to csv.')
    return os.path.join(out_csv_path)


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


def split(df, group):
    data = collections.namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, num_class):
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
        classes.append(dict_functions[num_class](row['class']))

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
    """
    Main func to
    :return:
    """
    parser: ArgumentParser = argparse.ArgumentParser(
        description="Convert images and images xml labels into tfrecords format",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-x', '--xml_path',
        help='Path to the folder where the image xml files stored. '
             'If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-c', '--output_dir',
        help='Path to dir there csv output and tfrecord file will appear',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-n', '--num_classes',
        help='Num classes of items 53 or 28',
        type=str,
        default='53'
    )
    parser.add_argument(
        '-i', '--image_dir',
        help='Path to directory with images according with their xml files',
        type=str,
        default=os.getcwd()
    )
    args = parser.parse_args()

    # Обработка xml в csv
    csv_file_name = process_to_csv(args.xml_path, args.output_dir)

    # Путь к папке с изображениям
    path_img = os.path.join(args.image_dir)

    # Считывание csv файла
    examples = pd.read_csv(csv_file_name, encoding='utf-8')
    tfrecord_output_num = len(
        [x for x in os.listdir(args.output_dir) if f'{os.path.basename(args.xml_path)}.record' in x])
    if tfrecord_output_num == 0:
        path_to_new_record_file = os.path.join(args.output_dir, f'{os.path.basename(args.xml_path)}.record')
    else:
        path_to_new_record_file = os.path.join(args.output_dir,
                                               f'{os.path.basename(args.xml_path)}{tfrecord_output_num}.record')
    writer = tf.python_io.TFRecordWriter(path_to_new_record_file)
    grouped = split(examples, 'filename')
    count = len(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, path_img, args.num_classes)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(path_to_new_record_file))
    print(f'Total objects - {count}')


if __name__ == "__main__":
    tf.app.run()
