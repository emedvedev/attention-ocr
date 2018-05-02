from __future__ import absolute_import

import os
import logging

import tensorflow as tf


class Exporter(object):
    def __init__(self, model):
        self.model = model

    def save(self, path, model_format):
        if model_format == "savedmodel":
            logging.info("Creating a SavedModel.")

            builder = tf.saved_model.builder.SavedModelBuilder(path)
            freezing_graph = self.model.sess.graph
            builder.add_meta_graph_and_variables(
                self.model.sess,
                ["serve"],
                signature_def_map={
                    'serving_default': tf.saved_model.signature_def_utils.predict_signature_def(
                        {'input': freezing_graph.get_tensor_by_name('input_image_as_bytes:0')},
                        {
                            'output': freezing_graph.get_tensor_by_name('prediction:0'),
                            'probability': freezing_graph.get_tensor_by_name('probability:0')
                        }
                    ),
                },
                clear_devices=True)

            builder.save()

            logging.info("Exported SavedModel into %s", path)

        elif model_format == "frozengraph":

            logging.info("Creating a frozen graph.")

            if not os.path.exists(path):
                os.makedirs(path)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                self.model.sess,
                self.model.sess.graph.as_graph_def(),
                ['prediction', 'probability'],
            )

            with tf.gfile.GFile(path + '/frozen_graph.pb', "wb") as outfile:
                outfile.write(output_graph_def.SerializeToString())

            logging.info("Exported as %s", path + '/frozen_graph.pb')
