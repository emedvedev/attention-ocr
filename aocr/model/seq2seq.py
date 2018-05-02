# Copyright 2015 Google Inc. All Rights Reserved.  #
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=invalid-name

"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
    https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
    - basic_rnn_seq2seq: The most basic RNN-RNN model.
    - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
    - embedding_rnn_seq2seq: The basic model with input embedding.
    - embedding_tied_rnn_seq2seq: The tied model with input embedding.
    - embedding_attention_seq2seq: Advanced model with input embedding and
            the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
    - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
        e.g., if you want to write a model that generates captions for images).
    - rnn_decoder: The basic decoder based on a pure RNN.
    - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
    - sequence_loss: Loss for a sequence model returning average log-perplexity.
    - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
        (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import tensorflow as tf

try:
    from tensorflow.contrib.rnn.python.ops import rnn_cell_impl
except ImportError:
    from tensorflow.python.ops import rnn_cell_impl

try:
    linear = rnn_cell_impl._linear  # pylint: disable=protected-access
except AttributeError:
    # pylint: disable=protected-access,no-name-in-module
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell
    linear = core_rnn_cell._linear


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

    Returns:
        A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(prev,
                                   output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function


def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=tf.float32, scope=None,
                      initial_state_attention=False, attn_num_hidden=128):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
            in order to generate i+1-th input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors of
                shape [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either the i-th element
                of decoder_inputs or loop_function(output {i-1}, i)) as follows.
                First, we run the cell on a combination of the input and previous
                attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).
            state: The state of each decoder cell the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: when num_heads is not positive, there are no inputs, or shapes
            of attention_states are not set.
    """
    # MODIFIED ADD START
    assert num_heads == 1, 'We only consider the case where num_heads=1!'
    # MODIFIED ADD END
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(scope or "attention_decoder"):
        batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size, attention_vec_size])
            hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(tf.get_variable("AttnV_%d" % a,
                                     [attention_vec_size]))

        state = initial_state

        # MODIFIED: return both context vector and attention weights
        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            # MODIFIED ADD START
            ss = None  # record attention weights
            # MODIFIED ADD END
            ds = []  # Results of attention reads will be stored here.
            for a in xrange(num_heads):
                with tf.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y), [2, 3])
                    a = tf.nn.softmax(s)
                    ss = a
                    # a = tf.Print(a, [a], message="a: ",summarize=30)
                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(
                        tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2]
                    )
                    ds.append(tf.reshape(d, [-1, attn_size]))
            # MODIFIED DELETED return ds
            # MODIFIED ADD START
            return ds, ss
            # MODIFIED ADD END

        outputs = []
        # MODIFIED ADD START
        attention_weights_history = []
        # MODIFIED ADD END
        prev = None
        batch_attn_size = tf.stack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype)
                 for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            # MODIFIED DELETED attns = attention(initial_state)
            # MODIFIED ADD START
            attns, attn_weights = attention(initial_state)
            attention_weights_history.append(attn_weights)
            # MODIFIED ADD END
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            # input_size = inp.get_shape().with_rank(2)[1]
            # todo: use input_size
            input_size = attn_num_hidden
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=True):
                    # MODIFIED DELETED attns = attention(state)
                    # MODIFIED ADD START
                    attns, attn_weights = attention(state)
                    # MODIFIED ADD END
            else:
                # MODIFIED DELETED attns = attention(state)
                # MODIFIED ADD START
                attns, attn_weights = attention(state)
                attention_weights_history.append(attn_weights)
                # MODIFIED ADD END

            with tf.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    # MODIFIED DELETED return outputs, state
    # MODIFIED ADD START
    return outputs, state, attention_weights_history
    # MODIFIED ADD END


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, embedding_size, num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=tf.float32, scope=None,
                                initial_state_attention=False,
                                attn_num_hidden=128):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
        decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function.
        num_symbols: Integer, how many symbols come into the embedding.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_size: Size of the output vectors; if None, use output_size.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_symbols] and B has shape
            [num_symbols]; if provided and feed_previous=True, each fed previous
            output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
            used (the "GO" symbol), and all other decoder inputs will be generated by:
                next = embedding_lookup(embedding, argmax(previous_output)),
            In effect, this implements a greedy decoder. It can also be used
            during training to emulate http://arxiv.org/abs/1506.03099.
            If False, decoder_inputs are used as given (the standard decoder case).
        update_embedding_for_previous: Boolean; if False and feed_previous=True,
            only the embedding for the first symbol of decoder_inputs (the "GO"
            symbol) will be updated by back propagation. Embeddings for the symbols
            generated from the decoder itself remain unchanged. This parameter has
            no effect if feed_previous=False.
        dtype: The dtype to use for the RNN initial states (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with tf.variable_scope(scope or "embedding_attention_decoder"):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",
                                        [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(
            emb_inp, initial_state, attention_states, cell, output_size=output_size,
            num_heads=num_heads, loop_function=loop_function,
            initial_state_attention=initial_state_attention, attn_num_hidden=attn_num_hidden)


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
        1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
        ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with tf.name_scope(name, "sequence_loss_by_example",
                       logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # todo(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = tf.reshape(target, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logit, labels=target)
            else:
                crossent = softmax_loss_function(logits=logit, labels=target)
            log_perp_list.append(crossent * weight)
        log_perps = tf.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = tf.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        average_across_batch: If set, divide the returned cost by the batch size.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
        A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
        ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    with tf.name_scope(name, "sequence_loss", logits + targets + weights):
        cost = tf.reduce_sum(sequence_loss_by_example(
            logits, targets, weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = tf.shape(targets[0])[0]
            return cost / tf.cast(batch_size, tf.float32)

        return cost


def model_with_buckets(encoder_inputs_tensor, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    Args:
        encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
        decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
        targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
        weights: List of 1D batch-sized float-Tensors to weight the targets.
        buckets: A list of pairs of (input size, output size) for each bucket.
        seq2seq: A sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        per_example_loss: Boolean. If set, the returned loss will be a batch-sized
            tensor of losses for each sequence in the batch. If unset, it will be
            a scalar with the averaged loss from all examples.
        name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
        A tuple of the form (outputs, losses), where:
            outputs: The outputs for each bucket. Its j'th element consists of a list
                of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
            losses: List of scalar Tensors, representing losses for each bucket, or,
                if per_example_loss is set, a list of 1D batch-sized float Tensors.

    Raises:
        ValueError: If length of encoder_inputsut, targets, or weights is smaller
            than the largest (last) bucket.
    """
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = [encoder_inputs_tensor] + decoder_inputs + targets + weights
    with tf.name_scope(name, "model_with_buckets", all_inputs):
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            bucket = buckets[0]
            encoder_inputs = tf.split(encoder_inputs_tensor, bucket[0], 0)
            encoder_inputs = [tf.squeeze(inp, squeeze_dims=[0]) for inp in encoder_inputs]
            bucket_outputs, attention_weights_history = seq2seq(encoder_inputs[:int(bucket[0])],
                                                                decoder_inputs[:int(bucket[1])],
                                                                int(bucket[0]))
            if per_example_loss:
                loss = sequence_loss_by_example(
                    bucket_outputs, targets[:int(bucket[1])], weights[:int(bucket[1])],
                    average_across_timesteps=True,
                    softmax_loss_function=softmax_loss_function)
            else:
                loss = sequence_loss(
                    bucket_outputs, targets[:int(bucket[1])], weights[:int(bucket[1])],
                    average_across_timesteps=True,
                    softmax_loss_function=softmax_loss_function)

    return bucket_outputs, loss, attention_weights_history
