# Attention-based OCR

Visual attention-based OCR model for image recognition with additional tools for creating TFRecords datasets and exporting the trained model with weights as a [SavedModel](https://www.tensorflow.org/api_docs/python/tf/saved_model) or a frozen graph.

## Acknowledgements

This project is based on a model by [Qi Guo](http://qiguo.ml) and [Yuntian Deng](https://github.com/da03). You can find the original model in the [da03/Attention-OCR](https://github.com/da03/Attention-OCR) repository.

## The model

Authors: [Qi Guo](http://qiguo.ml) and [Yuntian Deng](https://github.com/da03).

The model first runs a sliding CNN on the image (images are resized to height 32 while preserving aspect ratio). Then an LSTM is stacked on top of the CNN. Finally, an attention model is used as a decoder for producing the final outputs.

![OCR example](http://cs.cmu.edu/~yuntiand/OCR-2.jpg)

## Installation

```
pip install aocr
```

Note: Tensorflow and Numpy will be installed as dependencies. Additional dependencies are `PIL`/`Pillow`, `distance`, and `six`.

Note #2: this project works with Tensorflow 1.x. Upgrade to Tensorflow 2 is planned, but if you want to help, please feel free to create a PR.

## Usage

### Create a dataset

To build a TFRecords dataset, you need a collection of images and an annotation file with their respective labels.

```
aocr dataset ./datasets/annotations-training.txt ./datasets/training.tfrecords
aocr dataset ./datasets/annotations-testing.txt ./datasets/testing.tfrecords
```

Annotations are simple text files containing the image paths (either absolute or relative to your working dir) and their corresponding labels:

```
datasets/images/hello.jpg hello
datasets/images/world.jpg world
```

### Train

```
aocr train ./datasets/training.tfrecords
```

A new model will be created, and the training will start. Note that it takes quite a long time to reach convergence, since we are training the CNN and attention model simultaneously.

The `--steps-per-checkpoint` parameter determines how often the model checkpoints will be saved (the default output dir is `checkpoints/`).

**Important:** there is a lot of available training options. See the CLI help or the `parameters` section of this README.

### Test and visualize

```
aocr test ./datasets/testing.tfrecords
```

Additionally, you can visualize the attention results during testing (saved to `out/` by default):

```
aocr test --visualize ./datasets/testing.tfrecords
```

Example output images in `results/correct`:

Image 0 (j/j):

![example image 0](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_0.jpg)

Image 1 (u/u):

![example image 1](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_1.jpg)

Image 2 (n/n):

![example image 2](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_2.jpg)

Image 3 (g/g):

![example image 3](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_3.jpg)

Image 4 (l/l):

![example image 4](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_4.jpg)

Image 5 (e/e):

![example image 5](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_5.jpg)

### Export

After the model is trained and a checkpoint is available, it can be exported as either a frozen graph or a SavedModel.

```bash

# SavedModel (default):
aocr export ./exported-model

# Frozen graph:
aocr export --format=frozengraph ./exported-model

```

Load weights from the latest checkpoints and export the model into the `./exported-model` directory.

**Note**: During training, it is possible to pass parameters describing the dimensions of the input images (`--max-width`, `--max-height`, etc.). If you used them during training, make sure to also pass them to the `export` command. Otherwise the exported model will not work properly when serving (next section).

### Serving

Exported SavedModel can be served as an HTTP REST API using [Tensorflow Serving](https://github.com/tensorflow/serving). You can start the server by running the following command:

```
tensorflow_model_server --port=9000 --rest_api_port=9001 --model_name=yourmodelname --model_base_path=./exported-model
```

**Note**: tensorflow_model_server requires a sub-directory with the version number to be present and inside it the files exported in the previous step. So you need to manually move contents of `exported-model` into `exported-model/1`.

Now you can send a prediction request to the running server, for example:

```
curl -X POST \
  http://localhost:9001/v1/models/aocr:predict \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{
  "signature_name": "serving_default",
  "inputs": {
     	"input": { "b64": "<your image encoded as base64>" }
  }
}'
```

REST API requires binary inputs to be encoded as Base64 and wrapped in an object containing a `b64` key. [See 'Encoding binary values' in Tensorflow Serving documentation](https://www.tensorflow.org/serving/api_rest#encoding_binary_values)



## Google Cloud ML Engine

To train the model in the [Google Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/), upload the training dataset into a Google Cloud Storage bucket and start a training job with the `gcloud` tool.

1. Set the environment variables:

```
# Prefix for the job name.
export JOB_PREFIX="aocr"

# Region to launch the training job in.
# Should be the same as the storage bucket region.
export REGION="us-central1"

# Your storage bucket.
export GS_BUCKET="gs://aocr-bucket"

# Path to store your training dataset in the bucket.
export DATASET_UPLOAD_PATH="training.tfrecords"
```

2. Upload the training dataset:

```
gsutil cp ./datasets/training.tfrecords $GS_BUCKET/$DATASET_UPLOAD_PATH
```

3. Launch the ML Engine job:

```
export NOW=$(date +"%Y%m%d_%H%M%S")
export JOB_NAME="$JOB_PREFIX$NOW"
export JOB_DIR="$GS_BUCKET/$JOB_NAME"

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir=$JOB_DIR \
    --module-name=aocr \
    --package-path=aocr \
    --region=$REGION \
    --scale-tier=BASIC_GPU \
    --runtime-version=1.2 \
    -- \
    train $GS_BUCKET/$DATASET_UPLOAD_PATH \
    --steps-per-checkpoint=500 \
    --batch-size=512 \
    --num-epoch=20
```

## Parameters

### Global

* `log-path`: Path for the log file.

### Testing

* `visualize`: Output the attention maps on the original image.

### Exporting

* `format`: Format for the export (either `savedmodel` or `frozengraph`).

### Training

* `steps-per-checkpoint`: Checkpointing (print perplexity, save model) per how many steps
* `num-epoch`: The number of whole data passes.
* `batch-size`: Batch size.
* `initial-learning-rate`: Initial learning rate, note the we use AdaDelta, so the initial value does not matter much.
* `target-embedding-size`: Embedding dimension for each target.
* `attn-num-hidden`: Number of hidden units in attention decoder cell.
* `attn-num-layers`: Number of layers in attention decoder cell. (Encoder number of hidden units will be `attn-num-hidden`*`attn-num-layers`).
* `no-resume`: Create new weights even if there are checkpoints present.
* `max-gradient-norm`: Clip gradients to this norm.
* `no-gradient-clipping`: Do not perform gradient clipping.
* `gpu-id`: GPU to use.
* `use-gru`: Use GRU cells instead of LSTM.
* `max-width`: Maximum width for the input images. WARNING: images with the width higher than maximum will be discarded.
* `max-height`: Maximum height for the input images.
* `max-prediction`: Maximum length of the predicted word/phrase.

## References

[Convert a formula to its LaTex source](https://github.com/harvardnlp/im2markup)

[What You Get Is What You See: A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938.pdf)

[Torch attention OCR](https://github.com/da03/torch-Attention-OCR)
