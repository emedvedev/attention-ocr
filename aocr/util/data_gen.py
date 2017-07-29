import numpy as np
import tensorflow as tf

from .bucketdata import BucketData


class DataGen(object):
    GO_ID = 1
    EOS_ID = 2
    IMAGE_HEIGHT = 32

    def __init__(self,
                 annotation_fn,
                 buckets,
                 epochs=1000):
        """
        :param annotation_fn:
        :param lexicon_fn:
        :param valid_target_len:
        :param img_width_range: only needed for training set
        :param word_len:
        :param epochs:
        :return:
        """
        self.epochs = epochs

        self.bucket_specs = buckets
        self.bucket_data = BucketData()

        dataset = tf.contrib.data.TFRecordDataset([annotation_fn])
        dataset = dataset.map(self._parse_record)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.repeat(self.epochs)

    def clear(self):
        self.bucket_data = BucketData()

    def gen(self, batch_size):

        dataset = self.dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        images, labels = iterator.get_next()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            while True:
                try:
                    raw_images, raw_labels = sess.run([images, labels])
                    for img, lex in zip(raw_images, raw_labels):
                        word = self.convert_lex(lex)

                        bucket_size = self.bucket_data.append(img, word, lex)
                        if bucket_size >= batch_size:
                            bucket = self.bucket_data.flush_out(
                                self.bucket_specs,
                                go_shift=1)
                            if bucket is not None:
                                yield bucket
                            else:
                                assert False, 'No valid bucket.'
                except tf.errors.OutOfRangeError:
                    break

        self.clear()

    def convert_lex(self, lex):
        assert lex and len(lex) < self.bucket_specs[-1][1]

        word = [self.GO_ID]
        for char in lex:
            assert 96 < ord(char) < 123 or 47 < ord(char) < 58
            word.append(
                ord(char) - 97 + 13 if ord(char) > 96 else ord(char) - 48 + 3)
        word.append(self.EOS_ID)
        word = np.array(word, dtype=np.int32)

        return word

    @staticmethod
    def _parse_record(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })
        return features["image"], features["label"]
