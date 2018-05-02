from __future__ import absolute_import

import numpy as np


class BucketData(object):
    def __init__(self):
        self.data_list = []
        self.label_list = []
        self.label_list_plain = []
        self.comment_list = []

    def append(self, datum, label, label_plain, comment):
        self.data_list.append(datum)
        self.label_list.append(label)
        self.label_list_plain.append(label_plain)
        self.comment_list.append(comment)

        return len(self.data_list)

    def flush_out(self, bucket_specs, valid_target_length=float('inf'),
                  go_shift=1):
        # print self.max_width, self.max_label_len
        res = {}

        decoder_input_len = bucket_specs[0][1]

        # ENCODER PART
        res['data'] = np.array(self.data_list)
        res['labels'] = self.label_list_plain
        res['comments'] = self.comment_list

        # DECODER PART
        target_weights = []
        for l_idx in range(len(self.label_list)):
            label_len = len(self.label_list[l_idx])
            if label_len <= decoder_input_len:
                self.label_list[l_idx] = np.concatenate((
                    self.label_list[l_idx],
                    np.zeros(decoder_input_len - label_len, dtype=np.int32)))
                one_mask_len = min(label_len - go_shift, valid_target_length)
                target_weights.append(np.concatenate((
                    np.ones(one_mask_len, dtype=np.float32),
                    np.zeros(decoder_input_len - one_mask_len,
                             dtype=np.float32))))
            else:
                raise NotImplementedError

        res['decoder_inputs'] = [a.astype(np.int32) for a in
                                 np.array(self.label_list).T]
        res['target_weights'] = [a.astype(np.float32) for a in
                                 np.array(target_weights).T]

        assert len(res['decoder_inputs']) == len(res['target_weights'])

        self.data_list, self.label_list, self.label_list_plain, self.comment_list = [], [], [], []

        return res

    def __len__(self):
        return len(self.data_list)

    def __iadd__(self, other):
        self.data_list += other.data_list
        self.label_list += other.label_list
        self.label_list_plain += other.label_list_plain
        self.comment_list += other.comment_list

    def __add__(self, other):
        res = BucketData()
        res.data_list = self.data_list + other.data_list
        res.label_list = self.label_list + other.label_list
        res.label_list_plain = self.label_list_plain + other.label_list_plain
        res.comment_list = self.comment_list + other.comment_list
        return res
