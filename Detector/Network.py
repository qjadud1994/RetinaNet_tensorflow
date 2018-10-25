import os
import abc
import Detector.layers as layers
import tensorflow as tf
from Detector.input_producer import InputProducer
#from text_recognizer.common import InputFeatures, dense_to_sparse, EOS_INDEX

FLAGS = tf.app.flags.FLAGS


class Network(abc.ABC):

    def __init__(self, loss_fn=None):
        self.valid_loss_fn = ['ctc_loss',
                              'cross_ent']
        self.loss_fn = loss_fn

    def preprocess_image(self, image, is_train=True):
        return image

    def get_input(self,
                  batch_size,
                  is_train=True,
                  num_gpus=1):
        input_features = []
        batch_size = batch_size // max(1, num_gpus)
        for gpu_indx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_indx):
                if is_train:
                    input_feature = self.get_train_input(batch_size,
                                                         '/gpu:%d' % gpu_indx)
                else:
                    input_feature = self.get_test_input(batch_size)
                input_features.append(input_feature)
        return input_features

    def get_train_input(self, batch_size, input_device):
        input_producer = InputProducer(self.out_charset,
                                       self.preprocess_image,
                                       FLAGS.vertical_image)
        image, width, label, length, text, filename = \
                input_producer.bucketed_input_pipeline(
                        FLAGS.train_path,
                        str.split(FLAGS.filename_pattern, ','),
                        batch_size=batch_size,
                        num_threads=FLAGS.num_input_threads,
                        input_device=input_device,
                        width_threshold=FLAGS.width_threshold,
                        length_threshold=FLAGS.length_threshold,
                        verbose=FLAGS.verbose)
        input_feature = InputFeatures(image,width,label,
                                      length,text,filename)
        return input_feature

    def get_test_input(self, batch_size):
        from copy import deepcopy
        resize_hw = deepcopy(FLAGS.resize_hw)
        if resize_hw[0] < 0 : resize_hw[0] = None
        if resize_hw[1] < 0 : resize_hw[1] = None
        if resize_hw[0] is not None : resize_hw[0] = resize_hw[0] + FLAGS.padding*2
        if resize_hw[1] is not None : resize_hw[1] = resize_hw[1] + FLAGS.padding*2
        image = tf.placeholder(tf.uint8,
                shape=[batch_size, resize_hw[0], resize_hw[1], 1], name='image')
        width = tf.placeholder(tf.int32, shape=[batch_size], name='width')
        label = tf.placeholder(tf.int32, name='label')
        length = tf.placeholder(tf.int64, shape=[batch_size], name='length')
        text = tf.placeholder(tf.string, shape=[batch_size], name='text')
        filename = tf.placeholder(tf.string, shape=[batch_size], name='filename')
        input_feature = InputFeatures(image,width,label,
                                      length,text,filename)
        return input_feature

    @abc.abstractmethod
    def get_logits(self,
                   image,
                   mode,
                   **kwargs):
        return

    def get_loss(self,
                 logits,
                 label,
                 **kwargs):
        if self.loss_fn == 'cross_ent':
            return self._get_cross_entropy(logits,
                                           label,
                                           kwargs['label_length'],
                                           kwargs['label_maxlen'])
        elif self.loss_fn == 'ctc_loss':
            return self._get_ctc_loss(logits,
                                      label,
                                      kwargs['sequence_length'])
        else:
            raise NotImplementedError
        return

    def get_prediction(self, logits, sequence_length=None):
        if self.loss_fn == 'cross_ent':
            return self._get_argmax_prediction(logits)
        elif self.loss_fn == 'ctc_loss':
            return self._get_ctc_prediction(logits, sequence_length)
        else:
            raise NotImplementedError
        return

    def _get_cross_entropy(self,
                           logits,
                           label,
                           label_length,
                           label_maxlen=25):
        with tf.name_scope("train"):
            scope = self.scope or FLAGS.tune_scope
            scope = '|'.join(['train_tower_[0-9]+/'+s for s in scope.split('|')])
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=scope)
            # Compute sequence loss
            batch_size = logits.get_shape()[0].value
            num_classes = len(self.out_charset)
            label = tf.sparse_to_dense(
                sparse_indices=label.indices,
                sparse_values=label.values,
                output_shape=[batch_size, label_maxlen],
                default_value=num_classes)
            loss = 0
            for i in range(label_maxlen):
                _logit = logits[:,i,:]
                _label = label[:,i]
                _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logit,labels=_label)
                _mask = tf.to_float(tf.greater_equal(label_length, i))
                _mask = tf.reshape(_mask, _loss.get_shape())
                loss += tf.reduce_sum(_loss*_mask)
            loss /= batch_size
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return tvars, loss, extra_update_ops

    def _get_ctc_loss(self,
                      logits,
                      label,
                      sequence_length):
        with tf.name_scope("train"):
            scope = self.scope or FLAGS.tune_scope
            scope = '|'.join(['train_tower_[0-9]+/'+s for s in scope.split('|')])
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=scope)
            loss = layers.ctc_loss_layer(logits, label, sequence_length)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return tvars, loss, extra_update_ops

    def _get_argmax_prediction(self, logits):
        num_classes = len(self.out_charset)
        batch_size, label_maxlen, _ = logits.get_shape()
        predictions = []
        is_valid = tf.ones([batch_size], dtype=tf.int64)
        for i in range(label_maxlen):
            _logit = logits[:,i,:]
            _pred = tf.argmax(_logit, axis=-1)
            _pred = _pred*is_valid + num_classes*(1-is_valid)
            predictions.append(_pred)
            is_valid *= tf.to_int64(tf.not_equal(_pred, EOS_INDEX))
        predictions = tf.stack(predictions, axis=1)
        predictions = dense_to_sparse(
                predictions,
                eos_token=num_classes)
        return predictions

    def _get_ctc_prediction(self, logits, sequence_length):
        predictions, _ = tf.nn.ctc_beam_search_decoder(logits,
                                                       sequence_length,
                                                       beam_width=5,
                                                       top_paths=1,
                                                       merge_repeated=False)
        return predictions[0]