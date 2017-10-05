import tensorflow as tf
import logging

from six import text_type


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path, log_step=5000):
    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)

    with open(annotations_path, 'r') as f:
        for idx, line in enumerate(f):
            (img_path, label) = line.rstrip('\n').split('\t', 1)
            with open(img_path, 'rb') as img_file:
                img = img_file.read()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(img),
                'label': _bytes_feature(text_type.encode(label))}))

            writer.write(example.SerializeToString())

            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)

    logging.info('Dataset is ready: %i pairs.', idx+1)

    writer.close()
