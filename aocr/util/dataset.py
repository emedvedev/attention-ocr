import tensorflow as tf
import logging

from six import b


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path, log_step=5000, force_uppercase=True, save_filename=False):
    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)

    longest_label = ''

    with open(annotations_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n')
            try:
                (img_path, label) = line.split('\t', 1)
            except ValueError:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue

            with open(img_path, 'rb') as img_file:
                img = img_file.read()

            if force_uppercase:
                label = label.upper()

            if len(label) > len(longest_label):
                longest_label = label

            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(b(label))
            if save_filename:
                feature['comment'] = _bytes_feature(b(img_path))

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)

    logging.info('Dataset is ready: %i pairs.', idx+1)
    logging.info('Longest label (%i): %s', len(longest_label), longest_label)

    writer.close()
