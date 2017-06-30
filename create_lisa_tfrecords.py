
"""
Code adapted and modified by Ajmal Moochingal
Original code credit goes to Tensorflow Authors. 

Convert the LISA Traffic Sign dataset into Tensorflow tfrecords.

1. Download LISA Dataset here : http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
2. Specify dataset root directory and Output directory

Example usage:
    ./create_lisa_tf_record --data_dir=/home/user/lisa \
        --output_dir=/home/user/lisa/output

"""

import csv
from PIL import Image

import hashlib
import io
import logging
import os
import random
import re

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw LISA dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/lisa_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

data_dir = ''

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: data corresponding to each image file.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
       dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = image_subdirectory
  with tf.gfile.GFile(img_path) as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'PNG':
    raise ValueError('Image format error')
    # bg = PIL.Image.new("RGB", image.size, (255,255,255))
    # x, y = image.size
    # bg.paste(image)
    # img_path = img_path[:-3] + 'jpg'
    # bg.save(img_path)
    # with tf.gfile.GFile(img_path) as fid:
    #   encoded_jpg = fid.read()
    #   encoded_jpg_io = io.BytesIO(encoded_jpg)
    #   image = PIL.Image.open(encoded_jpg_io)
    #   if image.format != 'JPEG':
    #     raise ValueError('sase')

  key = hashlib.sha256(encoded_jpg).hexdigest()

  width, height = image.size
  img_filename = img_path.split('/')[-1]
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  occlud = []
  
  xmin.append(int(data[2]) / width)
  ymin.append(int(data[3]) / height)
  xmax.append(int(data[4]) / width)
  ymax.append(int(data[5]) / height)
  class_name = data[1]
  classes_text.append(class_name)
  classes.append(label_map_dict[class_name])

  trun, occ = data[6].split(',')
  truncated.append(int(trun))
  occlud.append(int(occ))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(img_filename),
      'image/source_id': dataset_util.bytes_feature(img_filename),
      'image/key/sha256': dataset_util.bytes_feature(key),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('png'),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.int64_list_feature(occlud),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
      print ('On image %d of %d'%( idx, len(examples)))
    example =  example[0]
    image_path = os.path.join(data_dir, example[0])

    if not os.path.exists(image_path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue

    tf_example = dict_to_tf_example(example, label_map_dict, image_path)
    writer.write(tf_example.SerializeToString())

  writer.close()


def main(_):
  
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from LISA dataset.')
  
  annotations_dir = os.path.join(data_dir, 'allAnnotations.csv')

  with open(annotations_dir) as csvFile :
    datareader = csv.reader(csvFile, delimiter = ';')
    next(datareader) # for skipping first row
    parse_data = []
    for row in datareader:
      parse_data.append([row])
  

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split. This happens randomly

  random.seed(49)
  random.shuffle(parse_data)
  num_examples = len(parse_data)

  num_train = int(0.9 * num_examples)
  train_examples = parse_data[:num_train]
  val_examples = parse_data[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'lisa_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'lisa_val.record')
  create_tf_record(train_output_path, label_map_dict, train_examples)
  create_tf_record(val_output_path, label_map_dict, val_examples)

if __name__ == '__main__':
  tf.app.run()
