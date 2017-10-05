import os
import tensorflow as tf
import logging


class Exporter(object):
    def __init__(self, checkpoint_dir):
        logging.info("Loading the checkpoint.")
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session(graph=graph)
            self.saver = tf.train.import_meta_graph(
                input_checkpoint + '.meta',
                clear_devices=True,
            )
            self.saver.restore(self.sess, input_checkpoint)
        logging.info("Loaded the checkpoint from %s", checkpoint_dir)

    def save(self, path, model_format):

        if model_format == "savedmodel":

            logging.info("Creating a SavedModel.")

            builder = tf.saved_model_builder.SavedModelBuilder(path)
            builder.add_meta_graph_and_variables(self.sess, [])
            builder.save()

            logging.info("Exported SavedModel into %s", path)

        elif model_format == "frozengraph":

            logging.info("Creating a frozen graph.")

            if not os.path.exists(path):
                os.makedirs(path)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                self.sess,
                tf.get_default_graph().as_graph_def(),
                ['prediction'],
            )
            with tf.gfile.GFile(path + '/frozen_graph.pb', "wb") as f:
                f.write(output_graph_def.SerializeToString())

            logging.info("Exported as %s", path + '/frozen_graph.pb')
