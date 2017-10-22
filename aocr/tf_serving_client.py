
# as of 2017-oct-16, tensorflow serving api only supports python 2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'Server host:port.')
tf.app.flags.DEFINE_string('model', 'attention-ocr',
                           'Model name.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'serving_default'

    if len(sys.argv) < 2:
        print('usage:')
        print(sys.argv[0], 'image_filename')
        sys.exit(1)

    img_path = sys.argv[1]
    with open(img_path, 'rb') as img_file:
        img = img_file.read()

    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img, shape=[1]))

    result_future = stub.Predict.future(request, 5.0)
    prediction = result_future.result().outputs['output'].string_val[0]

    print(prediction)

if __name__ == '__main__':
    tf.app.run()
