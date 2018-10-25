import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import cv2, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from Detector.RetinaNet import RetinaNet
from utils.bbox import draw_bboxes

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.WARN)
tf.app.flags.DEFINE_string('f', '', 'kernel')
#### Input pipeline
tf.app.flags.DEFINE_integer('input_size', 608,
                            """Input size""")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Train batch size""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                            """Learninig rate""")
tf.app.flags.DEFINE_integer('num_input_threads', 4,
                            """Number of readers for input data""")
tf.app.flags.DEFINE_integer('num_classes', 20,
                            """number of classes""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """The number of gpu""")
tf.app.flags.DEFINE_string('tune_from', 'results_test1/model.ckpt-70000',
                           """Path to pre-trained model checkpoint""")
#tf.app.flags.DEFINE_string('tune_from', 'results_test1/',
#                           """Path to pre-trained model checkpoint""")


#### Train dataset
tf.app.flags.DEFINE_string('train_path', '../data/mjsynth/train',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern', '*.tfrecord',
                           """File pattern for input data""")
### Validation dataset (during training)
tf.app.flags.DEFINE_string('valid_dataset','VOC',
                          """Validation dataset name""")
tf.app.flags.DEFINE_integer('valid_device', 0,
                           """Device for validation""")
tf.app.flags.DEFINE_integer('valid_batch_size', 8,
                            """Validation batch size""")
tf.app.flags.DEFINE_boolean('use_validation', True,
                            """Whether use validation or not""")
tf.app.flags.DEFINE_integer('valid_steps', 1000,
                            """Validation steps""")

#### Output Path
tf.app.flags.DEFINE_string('output', 'results_test2/model.ckpt-66000',
                           """Directory for event logs and checkpoints""")
#### Training config
tf.app.flags.DEFINE_float('cls_thresh', 0.5,
                            """thresh for class""")
tf.app.flags.DEFINE_float('nms_thresh', 0.5,
                            """thresh for nms""")
tf.app.flags.DEFINE_integer('max_detect', 300,
                            """num of max detect (using in nms)""")
tf.app.flags.DEFINE_string('tune_scope', '',
                           """Variable scope for training""")
tf.app.flags.DEFINE_integer('max_num_steps', 2**21,
                            """Number of optimization steps to run""")
tf.app.flags.DEFINE_boolean('verbose', True,
                            """Print log in tensorboard""")
tf.app.flags.DEFINE_boolean('use_profile', False,
                            """Whether use Tensorflow Profiling""")
tf.app.flags.DEFINE_boolean('use_debug', False,
                            """Whether use TFDBG or not""")
tf.app.flags.DEFINE_integer('save_steps', 2000,
                            """Save steps""")
tf.app.flags.DEFINE_integer('summary_steps', 50,
                            """Save steps""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                            """Moving Average dacay factor""")

img_dir = "/root/DB/VOC/VOC2012/JPEGImages/"

def _get_init_pretrained(sess):
    saver_reader = tf.train.Saver(tf.global_variables())
    saver_reader.restore(sess, FLAGS.tune_from)

with tf.Graph().as_default():
    image = tf.placeholder(tf.float32, shape=[1, 608, 608, 3], name='image')

    with tf.variable_scope('train_tower_0') as scope:
        net = RetinaNet("resnet50")

        box_head, cls_head = net.get_logits(image, True)

        decode = net.decode(box_head, cls_head)

    init_op = tf.group( tf.global_variables_initializer(),
                        tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        _get_init_pretrained(sess)

        for n, _img in enumerate(os.listdir(img_dir)):
            img = cv2.imread(img_dir + _img)
            img = cv2.resize(img, (608, 608))

            batch_image = np.expand_dims(img, 0)

            box, label = sess.run(decode, feed_dict={image : batch_image})

            img = draw_bboxes(img, box, label)
            #plt.imshow(img)
            print(img.shape)
            if n==1:
                break
