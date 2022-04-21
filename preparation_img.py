"""
Convert images and images xml labels into tfrecords format

optional arguments:
  -h, --help            show this help message and exit
  -x XML_PATH, --xml_path XML_PATH
                        Path to the folder where the image xml files stored. If not specified, the CWD will be used.
  -c OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to dir where the csv file's will be store. If not specified, the CWD will be used.

"""

import os
import argparse
import sys
from argparse import ArgumentParser

from xml_to_csv import xml_to_csv


# print("""
# Enter dir\'s path with XML files from LabelImg
# It may be train and test dirs.
# ├── XML
# │    ├── test
# │    └── train
# """)
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
    for directory in os.listdir(image_xml_path):
        dir_path = os.path.join(image_xml_path, directory)
        xml_df = xml_to_csv(dir_path)
        if output_dir is not None and os.path.exists(output_dir) and os.path.isdir(output_dir):
            xml_df.to_csv(os.path.join(output_dir, f'{directory}_objects_labels.csv'), index=None, encoding='utf-8')
        else:
            sys.exit()
        print(f'Successfully converted dir \"{directory}\" with xml to csv.')


def main():
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
        help='Path to dir where the csv file\'s will be store. '
             'If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    args = parser.parse_args()
    process_to_csv(args.xml_path, args.output_dir)


if __name__ == "__main__":
    main()
