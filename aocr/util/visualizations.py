from __future__ import absolute_import
from __future__ import division

import math
import os

from io import BytesIO

import numpy as np

from PIL import Image


def visualize_attention(filename, output_dir, attentions, pred, pad_width,
                        pad_height, threshold=1, normalize=False,
                        binarize=True, ground=None, flag=None):
    """Visualize the focus of the attention mechanism on an image.

    Parameters
    ----------
    filename : string
        Input filename.
    output_dir : string
        Output directory for visualizations.
    attentions : array of shape [len(pred), attention_size]
        Attention weights.
    pred : string
        Predicted output.
    pad_width : int
        Padded image width in pixels used as model input.
    pad_height : int
        Padded image height in pixels used as model input.
    threshold : int or float, optional (default=1)
        Threshold of maximum attention weight to display.
    normalize : bool, optional (default=False)
        Normalize the attention values to the [0, 1] range.
    binarize : bool, optional (default=True)
        If normalized, set attention values below `threshold` to 0.
        If not normalized, set maximum attention values to 1 and others to 0.
    ground : string or None, optional (default=None)
        Ground truth label.
    flag : bool or None, optional (default=None)
        Incorrect prediction flag.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if flag is None:
        filestring = 'predict-{}'.format(str(pred))
        idx = 2
        while filestring in os.listdir(output_dir):
            filestring = 'predict-{}-{}'.format(str(pred), idx)
            idx += 1
        out_dir = output_dir
    elif flag:
        filestring = os.path.splitext(os.path.basename(filename))[0]
        out_dir = os.path.join(output_dir, 'incorrect')
    else:
        filestring = os.path.splitext(os.path.basename(filename))[0]
        out_dir = os.path.join(output_dir, 'correct')
    out_dir = os.path.join(out_dir, filestring.replace('/', '_'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'word.txt'), 'w') as fword:
        fword.write(pred + '\n')
        if ground is not None:
            fword.write(ground)

        if isinstance(filename, str):
            img_file = open(filename, 'rb')
            img = Image.open(img_file)
        else:
            img = Image.open(BytesIO(filename))

        # Get image sequence with attention applied.
        img_data = np.asarray(img, dtype=np.uint8)
        img_out_frames, _ = map_attentions(img_data,
                                           attentions,
                                           pred,
                                           pad_width,
                                           pad_height,
                                           threshold=threshold,
                                           normalize=normalize,
                                           binarize=binarize)

        # Create initial image frame.
        img_out_init = (img_data * 0.3).astype(np.uint8)
        img_out_init = Image.fromarray(img_out_init)
        img_out_init = img_out_init.convert('RGB')

        # Assemble animation frames.
        img_out_frames = [img_out_init] + img_out_frames

        # Save cropped and animated images.
        output_animation = os.path.join(out_dir, 'image.gif')

        img_out_frames[0].save(output_animation, format='gif', save_all=True, loop=999,
                               duration=500, append_images=img_out_frames[1:])

        if isinstance(filename, str):
            img_file.close()
        img.close()


def map_attentions(img_data, attentions, pred, pad_width, pad_height,
                   threshold=1, normalize=False, binarize=True):
    """Map the attentions to the image."""
    img_out_agg = np.zeros(img_data.shape)
    img_out_frames = []

    width, height = img_data.shape[1], img_data.shape[0]

    # Calculate the image resizing proportions.
    width_resize_ratio, height_resize_ratio = 1, 1
    max_width = math.ceil((width / height) * pad_height)
    max_height = math.ceil((pad_width / max_width) * pad_height)
    if pad_width >= max_width:
        if pad_height < height:
            width_resize_ratio = width / max_width
            height_resize_ratio = height / pad_height
    else:
        width_resize_ratio = width / pad_width
        height_resize_ratio = height / max_height

    # Map the attention for each predicted character.
    for idx in range(len(pred)):
        attention = attentions[0][idx]

        # Get maximal attentional focus.
        score = attention.max()

        # Reshape the attention vector.
        nrows = 1  # should match number of encoded rows
        attention = attention.reshape((nrows, -1))

        # Map attention to fixed value.
        if normalize:
            attention *= (1.0 / attention.max())
            if binarize:
                attention[attention < threshold] = 0
        elif binarize:
            attention[attention >= score * threshold] = 1
            attention[attention < score] = 0

        # Resize attention to the image size, cropping padded regions.
        attention = Image.fromarray(attention)
        attention = attention.resize(
            (int(pad_width*width_resize_ratio), int(pad_height*height_resize_ratio)),
            Image.NEAREST)
        attention = attention.crop((0, 0, width, height))
        attention = np.asarray(attention)

        # Add new axis as needed (e.g., RGB images).
        if len(img_data.shape) == 3:
            attention = attention[..., np.newaxis]

        # Update the image frame with attended region(s).
        img_out_i = (img_data * np.maximum(attention, 0.3)).astype(np.uint8)
        img_out_i = Image.fromarray(img_out_i)
        img_out_i = img_out_i.convert('RGB')

        # Add animation frame to list.
        img_out_frames.append(img_out_i)

        # Add attention to aggregate.
        img_out_agg += attention

    return img_out_frames, img_out_agg
