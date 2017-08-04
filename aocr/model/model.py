"""Visual Attention Based OCR Model."""

from __future__ import absolute_import
from __future__ import division

import time
import os
import math
import logging

import distance
import numpy as np
import tensorflow as tf

from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin

from .cnn import CNN
from .seq2seq_model import Seq2SeqModel
from ..util.data_gen import DataGen


class Model(object):
    SYMBOLS = [''] * 3 + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __init__(self,
                 phase,
                 visualize,
                 data_path,
                 output_dir,
                 batch_size,
                 initial_learning_rate,
                 num_epoch,
                 steps_per_checkpoint,
                 target_vocab_size,
                 model_dir,
                 target_embedding_size,
                 attn_num_hidden,
                 attn_num_layers,
                 clip_gradients,
                 max_gradient_norm,
                 session,
                 load_model,
                 gpu_id,
                 use_gru,
                 max_image_width=160,
                 max_image_height=60,
                 max_prediction_length=8,
                 reg_val=0):

        # We need resized width, not the actual width
        max_image_width = int(math.ceil(1. * max_image_width / max_image_height * DataGen.IMAGE_HEIGHT))

        self.encoder_size = int(math.ceil(1. * max_image_width / 4))
        self.decoder_size = max_prediction_length + 2
        self.buckets = [(self.encoder_size, self.decoder_size)]

        gpu_device_id = '/gpu:' + str(gpu_id)
        self.gpu_device_id = gpu_device_id
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logging.info('loading data')
        # load data
        if phase == 'train':
            self.s_gen = DataGen(data_path, self.buckets, epochs=num_epoch)
        else:
            batch_size = 1
            self.s_gen = DataGen(data_path, self.buckets, epochs=1)

        logging.info('phase: %s' % phase)
        logging.info('model_dir: %s' % (model_dir))
        logging.info('load_model: %s' % (load_model))
        logging.info('output_dir: %s' % (output_dir))
        logging.info('steps_per_checkpoint: %d' % (steps_per_checkpoint))
        logging.info('batch_size: %d' % (batch_size))
        logging.info('num_epoch: %d' % num_epoch)
        logging.info('learning_rate: %d' % initial_learning_rate)
        logging.info('reg_val: %d' % (reg_val))
        logging.info('max_gradient_norm: %f' % max_gradient_norm)
        logging.info('clip_gradients: %s' % clip_gradients)
        logging.info('max_image_width %f' % max_image_width)
        logging.info('max_prediction_length %f' % max_prediction_length)
        logging.info('target_vocab_size: %d' % target_vocab_size)
        logging.info('target_embedding_size: %f' % target_embedding_size)
        logging.info('attn_num_hidden: %d' % attn_num_hidden)
        logging.info('attn_num_layers: %d' % attn_num_layers)
        logging.info('visualize: %s' % visualize)

        if use_gru:
            logging.info('using GRU in the decoder.')

        self.reg_val = reg_val
        self.sess = session
        self.steps_per_checkpoint = steps_per_checkpoint
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.global_step = tf.Variable(0, trainable=False)
        self.phase = phase
        self.visualize = visualize
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients

        if phase == 'train':
            self.forward_only = False
        elif phase == 'test':
            self.forward_only = True
        else:
            assert False, phase

        with tf.device(gpu_device_id):

            self.height = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.float64)
            self.max_width = max_image_width

            self.img_pl = tf.placeholder(tf.string, name='input_image_as_bytes')
            self.img_data = tf.cond(
                tf.less(tf.rank(self.img_pl), 1),
                lambda: tf.expand_dims(self.img_pl, 0),
                lambda: self.img_pl
            )
            self.img_data = tf.map_fn(self._prepare_image, self.img_data, dtype=tf.float32)
            num_images = tf.shape(self.img_data)[0]

            self.encoder_masks = []
            for i in xrange(self.encoder_size + 1):
                self.encoder_masks.append(
                    tf.tile([[1.]], [num_images, 1])
                )

            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(self.decoder_size + 1):
                self.decoder_inputs.append(
                    tf.tile([0], [num_images])
                )
                if i < self.decoder_size:
                    self.target_weights.append(tf.tile([1.], [num_images]))
                else:
                    self.target_weights.append(tf.tile([0.], [num_images]))

            # not 2, 2 is static (???)

            self.zero_paddings = tf.zeros([num_images, 2, 512], dtype=np.float32)

            cnn_model = CNN(self.img_data, True)
            self.conv_output = cnn_model.tf_output()
            self.concat_conv_output = tf.concat(axis=1, values=[self.conv_output, self.zero_paddings])
            self.perm_conv_output = tf.transpose(self.concat_conv_output, perm=[1, 0, 2])
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks=self.encoder_masks,
                encoder_inputs_tensor=self.perm_conv_output,
                decoder_inputs=self.decoder_inputs,
                target_weights=self.target_weights,
                target_vocab_size=target_vocab_size,
                buckets=self.buckets,
                target_embedding_size=target_embedding_size,
                attn_num_layers=attn_num_layers,
                attn_num_hidden=attn_num_hidden,
                forward_only=self.forward_only,
                use_gru=use_gru)

            ###
            # self.img_pl = tf.placeholder(tf.string, name='input_image_as_bytes')
            # self.img_data = tf.cond(
            #     tf.less(tf.rank(self.img_pl), 1),
            #     lambda: tf.expand_dims(self.img_pl, 0),
            #     lambda: self.img_pl
            # )
            # self.img_data = tf.map_fn(lambda x: tf.image.decode_png(x, channels=1), self.img_data, dtype=tf.uint8)

            # self.dims = tf.shape(self.img_data)
            # height_const = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.float64)
            # new_width = tf.to_int32(tf.ceil(tf.truediv(self.dims[2], self.dims[1]) * height_const))
            # self.new_dims = [tf.to_int32(height_const), new_width]  # [32, 85]  #

            # self.img_data = tf.image.resize_images(self.img_data, self.new_dims, method=tf.image.ResizeMethod.BICUBIC)

            # # variables

            # num_images = self.dims[0]
            # real_len = tf.to_int32(tf.maximum(tf.floor_div(tf.to_float(new_width), 4) - 1, 0))
            # padd_len = self.encoder_size - real_len

            # self.zero_paddings = tf.zeros([num_images, padd_len, 512], dtype=np.float32)

            # self.encoder_masks = []
            # for i in xrange(self.encoder_size + 1):
            #     self.encoder_masks.append(
            #         tf.cond(
            #             tf.less_equal(i, real_len),
            #             lambda: tf.tile([[1.]], [num_images, 1]),
            #             lambda: tf.tile([[0.]], [num_images, 1]),
            #         )
            #     )

            # self.decoder_inputs = []
            # self.target_weights = []
            # for i in xrange(self.decoder_size + 1):
            #     self.decoder_inputs.append(
            #         tf.tile([0], [num_images])
            #     )
            #     if i < self.decoder_size:
            #         self.target_weights.append(tf.tile([1.], [num_images]))
            #     else:
            #         self.target_weights.append(tf.tile([0.], [num_images]))

            # print self.img_data.get_shape()

            # cnn_model = CNN(self.img_data, True)
            # self.conv_output = cnn_model.tf_output()
            # self.concat_conv_output = tf.concat(axis=1, values=[self.conv_output, self.zero_paddings])
            # self.perm_conv_output = tf.transpose(self.concat_conv_output, perm=[1, 0, 2])
            # print self.perm_conv_output.get_shape()
            # self.attention_decoder_model = Seq2SeqModel(
            #     encoder_masks=self.encoder_masks,
            #     encoder_inputs_tensor=self.perm_conv_output,
            #     decoder_inputs=self.decoder_inputs,
            #     target_weights=self.target_weights,
            #     target_vocab_size=target_vocab_size,
            #     buckets=self.buckets,
            #     target_embedding_size=target_embedding_size,
            #     attn_num_layers=attn_num_layers,
            #     attn_num_hidden=attn_num_hidden,
            #     forward_only=self.forward_only,
            #     use_gru=use_gru)
            ###

            table = tf.contrib.lookup.MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value="",
                checkpoint=True,
            )

            insert = table.insert(
                tf.constant([i for i in xrange(len(self.SYMBOLS))], dtype=tf.int64),
                tf.constant(list(self.SYMBOLS)),
            )

            with tf.control_dependencies([insert]):

                output_feed = []

                for l in xrange(len(self.attention_decoder_model.output)):
                    guess = tf.argmax(self.attention_decoder_model.output[l], axis=1)
                    output_feed.append(table.lookup(guess))

                arr_prediction = tf.foldl(lambda a, x: a + x, output_feed)

                self.prediction = tf.cond(
                    tf.equal(tf.shape(arr_prediction)[0], 1),
                    lambda: arr_prediction[0],
                    lambda: arr_prediction
                )

            if not self.forward_only:  # train
                self.updates = []
                self.summaries_by_bucket = []

                params = tf.trainable_variables()
                opt = tf.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)

                if self.reg_val > 0:
                    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    logging.info('Adding %s regularization losses', len(reg_losses))
                    logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
                    loss_op = self.reg_val * tf.reduce_sum(reg_losses) + self.attention_decoder_model.loss
                else:
                    loss_op = self.attention_decoder_model.loss

                gradients, params = zip(*opt.compute_gradients(loss_op, params))
                if self.clip_gradients:
                    gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                # Add summaries for loss, variables, gradients, gradient norms and total gradient norm.
                summaries = []
                summaries.append(tf.summary.scalar("loss", loss_op))
                summaries.append(tf.summary.scalar("total_gradient_norm", tf.global_norm(gradients)))
                all_summaries = tf.summary.merge(summaries)
                self.summaries_by_bucket.append(all_summaries)
                # update op - apply gradients
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.updates.append(opt.apply_gradients(zip(gradients, params), global_step=self.global_step))


        self.saver_all = tf.train.Saver(tf.all_variables())
        self.checkpoint_path = os.path.join(self.model_dir, "model.ckpt")

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and load_model:
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())

    def test(self):
        loss = 0.0
        current_step = 0
        num_correct = 0.0
        num_total = 0.0

        for batch in self.s_gen.gen(1):
            current_step += 1
            # Get a batch and make a step.
            start_time = time.time()
            result = self.step(batch, self.forward_only)
            loss += result['loss'] / self.steps_per_checkpoint
            curr_step_time = (time.time() - start_time)

            if self.visualize:
                step_attns = np.array([[a.tolist() for a in step_attn] for step_attn in result['attentions']]).transpose([1, 0, 2])

            num_total += 1

            output = result['prediction']
            ground = batch['labels'][0]

            num_incorrect = 1 if output != ground else 0
            num_correct += 1.0 - num_incorrect

            if self.visualize:
                self.visualize_attention(batch['labels'][0], step_attns[0], output, ground, num_incorrect)

            correctness = "correct" if num_incorrect is 0 else "incorrect (%s vs %s)" % (output, ground)

            accuracy = num_correct / num_total * 100
            logging.info('Step %i (%.3fs): %s. Accuracy: %.2f, loss: %f, perplexity: %f.'
                         % (current_step,
                            curr_step_time,
                            correctness,
                            accuracy,
                            result['loss'],
                            math.exp(result['loss']) if result['loss'] < 300 else float('inf')))

    def train(self):
        step_time = 0.0
        loss = 0.0
        current_step = 0
        writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)

        logging.info('Starting the training process.')
        for batch in self.s_gen.gen(self.batch_size):

            current_step += 1

            start_time = time.time()
            result = self.step(batch, self.forward_only)
            loss += result['loss'] / self.steps_per_checkpoint
            curr_step_time = (time.time() - start_time)
            step_time += curr_step_time / self.steps_per_checkpoint

            num_correct = 0

            step_outputs = result['prediction']
            grounds = batch['labels']
            for output, ground in zip(step_outputs, grounds):
                num_incorrect = distance.levenshtein(output, ground)
                num_incorrect = float(num_incorrect) / len(ground)
                num_incorrect = min(1.0, num_incorrect)
                num_correct += 1. - num_incorrect

            writer.add_summary(result['summaries'], current_step)

            precision = num_correct / self.batch_size
            step_perplexity = math.exp(result['loss']) if result['loss'] < 300 else float('inf')

            logging.info('Step %i: %.3fs, precision: %.2f, loss: %f, perplexity: %f.'
                         % (current_step, curr_step_time, precision*100, result['loss'], step_perplexity))

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % self.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                # Print statistics for the previous epoch.
                logging.info("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f."
                             % (self.sess.run(self.global_step), step_time, loss, perplexity))
                # Save checkpoint and reset timer and loss.
                logging.info("Saving the model at step %d."%current_step)
                self.saver_all.save(self.sess, self.checkpoint_path, global_step=self.global_step)
                step_time, loss = 0.0, 0.0

        # Print statistics for the previous epoch.
        logging.info("Global step %d. Time: %.3f, loss: %f, perplexity: %.2f."
                     % (self.sess.run(self.global_step), step_time, loss, perplexity))
        # Save checkpoint and reset timer and loss.
        logging.info("Finishing the training and saving the model at step %d." % current_step)
        self.saver_all.save(self.sess, self.checkpoint_path, global_step=self.global_step)

    def to_savedmodel(self):
        raise NotImplementedError

    def to_frozengraph(self):
        raise NotImplementedError

    # step, read one batch, generate gradients
    def step(self, batch, forward_only):
        img_data = batch['data']
        decoder_inputs = batch['decoder_inputs']
        target_weights = batch['target_weights']

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_pl.name] = img_data

        if not forward_only:  # Train
            for l in xrange(self.decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]

            # Since our targets are decoder inputs shifted by one, we need one more.
            last_target = self.decoder_inputs[self.decoder_size].name
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        output_feed = [
            self.attention_decoder_model.loss,  # Loss for this batch.
            self.prediction
        ]

        if not forward_only:
            output_feed += [self.summaries_by_bucket[0],
                            self.updates[0]]
        elif self.visualize:
            output_feed += self.attention_decoder_model.attention_weights_history

        outputs = self.sess.run(output_feed, input_feed)

        res = {
            'loss': outputs[0],
            'prediction': outputs[1],
        }

        if not forward_only:
            res['summaries'] = outputs[2]
        elif self.visualize:
            res['attentions'] = outputs[2:]

        return res

    def visualize_attention(self, filename, attentions, output, label, flag_incorrect):
        if flag_incorrect:
            output_dir = os.path.join(self.output_dir, 'incorrect')
        else:
            output_dir = os.path.join(self.output_dir, 'correct')
        output_dir = os.path.join(output_dir, filename.replace('/', '_'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'word.txt'), 'w') as fword:
            fword.write(output+'\n')
            fword.write(label)
            with open(filename, 'rb') as img_file:
                img = Image.open(img_file)
                w, h = img.size
                mh = 32
                mw = math.floor(1. * w / h * mh)
                img = img.resize(
                        (mw, h),
                        Image.ANTIALIAS)
                img_data = np.asarray(img, dtype=np.uint8)
                for idx in range(len(output)):
                    output_filename = os.path.join(output_dir, 'image_%d.jpg' % (idx))
                    attention = attentions[idx][:(int(mw/4)-1)]
                    attention_orig = np.zeros(mw)
                    for i in range(mw):
                        if i/4-1 > 0 and i/4-1 < len(attention):
                            attention_orig[i] = attention[int(i/4)-1]
                    attention_orig = np.convolve(attention_orig, [0.199547, 0.200226, 0.200454, 0.200226, 0.199547], mode='same')
                    attention_orig = np.maximum(attention_orig, 0.3)
                    attention_out = np.zeros((h, mw))
                    for i in range(mw):
                        attention_out[:, i] = attention_orig[i]
                    if len(img_data.shape) == 3:
                        attention_out = attention_out[:, :, np.newaxis]
                    img_out_data = img_data * attention_out
                    img_out = Image.fromarray(img_out_data.astype(np.uint8))
                    img_out.save(output_filename)

    def _prepare_image(self, img):
        image = tf.image.decode_png(img, channels=1)
        dims = tf.shape(image)

        width = tf.to_int32(tf.ceil(tf.truediv(dims[1], dims[0]) * self.height))
        height = tf.to_int32(self.height)

        resized = tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BICUBIC)
        padded = tf.image.pad_to_bounding_box(resized, 0, 0, height, self.max_width)

        return padded
