import os
import tensorflow as tf
import logging


class Exporter(object):
    def __init__(self, checkpoint_dir):
        logging.info("Loading the checkpoint.")
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.saver = tf.train.import_meta_graph(
                input_checkpoint + '.meta',
                clear_devices=True,
            )
            self.saver.restore(self.sess, input_checkpoint)

            self.output_graph_def = tf.graph_util.convert_variables_to_constants(
                self.sess,
                tf.get_default_graph().as_graph_def(),
                ['prediction'],
            )
        logging.info("Loaded the checkpoint from %s", checkpoint_dir)

    def save(self, path, model_format):

        if model_format == "savedmodel":

            logging.info("Creating a SavedModel.")

            with tf.Graph().as_default() as freezing_graph, tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as freezing_sess:

                tf.import_graph_def(
                    self.output_graph_def,
                    input_map=None,
                    return_elements=None,
                    op_dict=None,
                    name="",
                    producer_op_list=None
                )

                builder = tf.saved_model.builder.SavedModelBuilder(path)
                builder.add_meta_graph_and_variables(freezing_sess,
                    ["serve"],
                    signature_def_map={
                        'serving_default': tf.saved_model.signature_def_utils.predict_signature_def(
                            {'input': freezing_graph.get_tensor_by_name('input_image_as_bytes:0')},
                            {'output': freezing_graph.get_tensor_by_name('prediction:0')}
                        ),
                    },
                    clear_devices=True)

                builder.save()

            logging.info("Exported SavedModel into %s", path)

        elif model_format == "frozengraph":

            logging.info("Creating a frozen graph.")

            if not os.path.exists(path):
                os.makedirs(path)

            with tf.gfile.GFile(path + '/frozen_graph.pb', "wb") as f:
                f.write(self.output_graph_def.SerializeToString())

            logging.info("Exported as %s", path + '/frozen_graph.pb')
