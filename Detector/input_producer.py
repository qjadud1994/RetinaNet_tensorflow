import os
import tensorflow as tf

slim = tf.contrib.slim

class InputProducer(object):

    def __init__(self, preprocess_image_fn=None, vertical_image=False):
        self.vertical_image = vertical_image
        self._preprocess_image = preprocess_image_fn if preprocess_image_fn is not None \
                                 else self._default_preprocess_image_fn

        self.ITEMS_TO_DESCRIPTIONS = {
            'image': 'A color image of varying height and width.',
            'shape': 'Shape of the image',
            'object/bbox': 'A list of bounding boxes, one per each object.',
            'object/label': 'A list of labels, one per each object.',
        }

        self.SPLITS_TO_SIZES = {
            'train': 9540,
            'val': 2000
        }
        #self.SPLITS_TO_SIZES = {
        #    'train_2000': 2000,
        #    'val_500': 500
        #}

        self.FILE_PATTERN = '%s.record'

    def num_classes(self):
        return 20

    def get_split(self, split_name, dataset_dir):
        """Gets a dataset tuple with instructions for reading Pascal VOC dataset.
        Args:
          split_name: A train/test split name.
          dataset_dir: The base directory of the dataset sources.
          file_pattern: The file pattern to use when matching the dataset sources.
            It is assumed that the pattern contains a '%s' string so that the split
            name can be inserted.
          reader: The TensorFlow reader type.
        Returns:
          A `Dataset` namedtuple.
        Raises:
            ValueError: if `split_name` is not a valid train/test split.
        """
        if split_name not in self.SPLITS_TO_SIZES:
            raise ValueError('split name %s was not recognized.' % split_name)

        file_pattern = os.path.join(dataset_dir, self.FILE_PATTERN % split_name)

        reader = tf.TFRecordReader

        # Features in Pascal VOC TFRecords.
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            #'image/height': tf.FixedLenFeature([1], tf.int64),
            #'image/width': tf.FixedLenFeature([1], tf.int64),
            #'image/channels': tf.FixedLenFeature([1], tf.int64),
            #'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
            #'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            #'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
        }
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            #'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
            #'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
            #'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        #if has_labels(dataset_dir):
        #    labels_to_names = read_label_file(dataset_dir)

        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=self.SPLITS_TO_SIZES[split_name],
            items_to_descriptions=self.ITEMS_TO_DESCRIPTIONS,
            num_classes=self.num_classes(),
            labels_to_names=labels_to_names)

    def _default_preprocess_image_fn(self, image, is_train=True):
        return image
