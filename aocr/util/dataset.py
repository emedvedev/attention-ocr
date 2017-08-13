import tensorflow as tf
import logging


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path, log_step=5000):
    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    with open(annotations_path, 'r') as f:
        pairs = [line.split() for line in f.readlines()]

    for idx, (img_path, label) in enumerate(pairs):

        with open(img_path, 'rb') as img_file:
            img = img_file.read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img),
            'label': _bytes_feature(label)}))

        writer.write(example.SerializeToString())

        if idx > 1 and idx % log_step == 0:
            logging.info('Processed %s pairs.', idx)

    logging.info('The dataset is ready: %i pairs.', len(pairs))

    writer.close()
