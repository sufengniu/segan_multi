from __future__ import print_function
import tensorflow as tf
import numpy as np
from model import SEAE
from model_multi import SEGAN_I, SEGAN_III, SEGAN_IV
import os
import os.path as osp
from tensorflow.python.client import device_lib
from scipy.io import wavfile
from data_loader import pre_emph

devices = device_lib.list_local_devices()

flags = tf.app.flags
flags.DEFINE_integer("seed",111, "Random seed (Def: 111).")
flags.DEFINE_integer("epoch", 150, "Epochs to train (Def: 150).")
flags.DEFINE_integer("batch_size", 150, "Batch size (Def: 150).")
flags.DEFINE_integer("save_freq", 500, "Batch save freq (Def: 500).")
flags.DEFINE_integer("canvas_size", 2**14, "Canvas size (Def: 2^14).")
flags.DEFINE_integer("denoise_epoch", 5, "Epoch where noise in disc is "
                                          "removed (Def: 5).")
flags.DEFINE_integer("l1_remove_epoch", 150, "Epoch where L1 in G is "
                                           "removed (Def: 150).")
flags.DEFINE_boolean("bias_deconv", False,
                     "Flag to specify if we bias deconvs (Def: False)")
flags.DEFINE_boolean("bias_downconv", False,
                     "flag to specify if we bias downconvs (def: false)")
flags.DEFINE_boolean("bias_D_conv", False,
                     "flag to specify if we bias D_convs (def: false)")

# TODO: noise decay is under check
flags.DEFINE_float("denoise_lbound", 0.01, "Min noise std to be still alive (Def: 0.001)")
flags.DEFINE_float("noise_decay", 0.7, "Decay rate of noise std (Def: 0.7)")
flags.DEFINE_float("d_label_smooth", 0.1, "Smooth factor in D (Def: 0.1)")
flags.DEFINE_float("init_noise_std", 0.5, "Init noise std (Def: 0.5)")
flags.DEFINE_float("init_l1_weight", 100., "Init L1 lambda (Def: 100)")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability in D, if is 1, no dropout used")

flags.DEFINE_integer("z_dim", 256, "Dimension of input noise to G (Def: 256).")
flags.DEFINE_integer("z_depth", 256, "Depth of input noise to G (Def: 256).")
flags.DEFINE_string("save_path", "/scratch3/sniu/segan/segan_results", "Path to save out model "
                                                   "files. (Def: dwavegan_model"
                                                   ").")
flags.DEFINE_string("g_nl", "leaky", "Type of nonlinearity in G: leaky or prelu. (Def: leaky).")
flags.DEFINE_string("model", "gan", "Type of model to train: gan or ae. (Def: gan).")
flags.DEFINE_string("deconv_type", "deconv", "Type of deconv method: deconv or "
                                             "nn_deconv (Def: deconv).")
flags.DEFINE_string("g_type", "ae", "Type of G to use: ae or dwave. (Def: ae).")
flags.DEFINE_string("decoder_type", "IV", "Type of decoder in G. I|II|III|IV|V. I: original SEGAN, II: ")
flags.DEFINE_float("g_learning_rate", 0.0001, "G learning_rate (Def: 0.0002)")
flags.DEFINE_float("d_learning_rate", 0.0001, "D learning_rate (Def: 0.0002)")
flags.DEFINE_float("beta_1", 0.5, "Adam beta 1 (Def: 0.5)")
flags.DEFINE_float("preemph", 0.95, "Pre-emph factor (Def: 0.95)")
flags.DEFINE_string("synthesis_path", "/scratch3/sniu/segan/dwavegan_samples", "Path to save output"
                                                          " generated samples."
                                                          " (Def: dwavegan_sam"
                                                          "ples).")
flags.DEFINE_string("e2e_dataset", "/scratch3/sniu/data/segan.tfrecords", "TFRecords"
                                                          " (Def: data/"
                                                          "segan.tfrecords.")
flags.DEFINE_string("save_clean_path", "/scratch3/sniu/segan/test_clean_results", "Path to save clean utts")
flags.DEFINE_string("test_dir", None, "name of test wav (it won't train)")
flags.DEFINE_string("weights", None, "Weights file")
FLAGS = flags.FLAGS

def pre_emph_test(coeff, canvas_size):
    x_ = tf.placeholder(tf.float32, shape=[canvas_size,])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph

def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def main(_):
    print('Parsed arguments: ', FLAGS.__flags)

    # make save path if it is required
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    if not os.path.exists(FLAGS.synthesis_path):
        os.makedirs(FLAGS.synthesis_path)
    np.random.seed(FLAGS.seed)
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    udevices = []
    for device in devices:
        if len(devices) > 1 and 'CPU' in device.name:
            # Use cpu only when we dont have gpus
            continue
        print('Using device: ', device.name)
        udevices.append(device.name)
    # execute the session
    with tf.Session(config=config) as sess:
        if FLAGS.model == 'gan':
            print('Creating GAN model')
            if FLAGS.decoder_type == 'I' or FLAGS.decoder_type == 'II':
                se_model = SEGAN_I(sess, FLAGS, udevices, name='SEGAN_'+FLAGS.decoder_type)
            elif FLAGS.decoder_type == 'III':
                se_model = SEGAN_III(sess, FLAGS, udevices, name='SEGAN_'+FLAGS.decoder_type)
            else:
                se_model = SEGAN_IV(sess, FLAGS, udevices, name='SEGAN_'+FLAGS.decoder_type)
        elif FLAGS.model == 'ae':
            print('Creating AE model')
            se_model = SEAE(sess, FLAGS, udevices)
        else:
            raise ValueError('{} model type not understood!'.format(FLAGS.model))
        if FLAGS.test_dir is None:
            se_model.train(FLAGS, udevices)
        else:
            test_dir = create_dir(FLAGS.test_dir)
            if FLAGS.weights is None:
                raise ValueError('weights must be specified!')
            print('Loading model weights...')
            se_model.load(FLAGS.save_path, FLAGS.weights)
            for test_wav in os.listdir(test_dir):
                fm, wav_data = wavfile.read(test_dir+test_wav)
                wavname = test_wav
                if fm != 16000:
                    raise ValueError('16kHz required! Test file is different')
                wave = (2./65535.) * (wav_data.astype(np.float32) - 32767) + 1.
                if FLAGS.preemph  > 0:
                    print('preemph test wave with {}'.format(FLAGS.preemph))
                    x_pholder, preemph_op = pre_emph_test(FLAGS.preemph, wave.shape[0])
                    wave = sess.run(preemph_op, feed_dict={x_pholder:wave})
                print('test wave shape: ', wave.shape)
                print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
                c_wave = se_model.clean(wave)
                print('c wave min:{}  max:{}'.format(np.min(c_wave), np.max(c_wave)))
                write_dir = create_dir(FLAGS.save_clean_path)
                wavfile.write(os.path.join(write_dir, wavname), 16000, c_wave)
                print('Done cleaning {} and saved '
                      'to {}'.format(test_wav,
                                     os.path.join(FLAGS.save_clean_path, wavname)))


if __name__ == '__main__':
    tf.app.run()
