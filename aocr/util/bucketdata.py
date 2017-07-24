import math
import numpy as np


class BucketData(object):
    def __init__(self):
        self.max_width = 0
        self.max_label_len = 0
        self.data_list = []
        self.data_len_list = []
        self.label_list = []
        self.label_list_plain = []

    def append(self, datum, width, label, label_plain):
        self.data_list.append(datum)
        self.data_len_list.append(int(math.floor(float(width) / 4)) - 1)
        self.label_list.append(label)
        self.label_list_plain.append(label_plain.upper())

        self.max_width = max(width, self.max_width)
        self.max_label_len = max(len(label), self.max_label_len)

        return len(self.data_list)

    def flush_out(self, bucket_specs, valid_target_length=float('inf'),
                  go_shift=1):
        # print self.max_width, self.max_label_len
        res = dict(bucket_id=None,
                   data=None, zero_paddings=None, encoder_mask=None,
                   decoder_inputs=None, target_weights=None)

        def get_bucket_id():
            for idx in range(0, len(bucket_specs)):
                if bucket_specs[idx][0] >= self.max_width / 4 - 1 \
                        and bucket_specs[idx][1] >= self.max_label_len:
                    return idx
            return None

        res['bucket_id'] = get_bucket_id()
        if res['bucket_id'] is None:
            self.data_list, self.data_len_list, self.label_list = [], [], []
            self.max_width, self.max_label_len = 0, 0
            return None

        _, decoder_input_len = bucket_specs[res['bucket_id']]

        # ENCODER PART
        res['data_len'] = [a.astype(np.int32) for a in
                           np.array(self.data_len_list)]
        res['data'] = np.array(self.data_list)
        res['real_len'] = self.max_width
        res['labels'] = self.label_list_plain

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
                # self.label_list[l_idx] = \
                # self.label_list[l_idx][:decoder_input_len]
                # target_weights.append([1]*decoder_input_len)

        res['decoder_inputs'] = [a.astype(np.int32) for a in
                                 np.array(self.label_list).T]
        res['target_weights'] = [a.astype(np.float32) for a in
                                 np.array(target_weights).T]

        assert len(res['decoder_inputs']) == len(res['target_weights'])

        self.data_list, self.label_list, self.label_list_plain = [], [], []
        self.max_width, self.max_label_len = 0, 0

        return res

    def __len__(self):
        return len(self.data_list)

    def __iadd__(self, other):
        self.data_list += other.data_list
        self.label_list += other.label_list
        self.max_label_len = max(self.max_label_len, other.max_label_len)
        self.max_width = max(self.max_width, other.max_width)

    def __add__(self, other):
        res = BucketData()
        res.data_list = self.data_list + other.data_list
        res.label_list = self.label_list + other.label_list
        res.max_width = max(self.max_width, other.max_width)
        res.max_label_len = max((self.max_label_len, other.max_label_len))
        return res
