from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from scipy.io import wavfile
from model import Model
from generator import *
from discriminator import *
import numpy as np
from data_loader import read_and_decode, de_emph
from bnorm import VBN
from ops import *
import timeit
import os


class SEGAN_I(Model):
    """ Speech Enhancement Generative Adversarial Network """
    def __init__(self, sess, args, devices, infer=False, name='SEGAN_I'):
        super(SEGAN_I, self).__init__(name)
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        if infer:
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        else:
            self.keep_prob = args.keep_prob
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.d_label_smooth = args.d_label_smooth
        self.decoder_type = args.decoder_type
        self.devices = devices
        self.z_dim = args.z_dim
        self.z_depth = args.z_depth
        # type of deconv
        self.deconv_type = args.deconv_type
        # specify if use biases or not
        self.bias_downconv = args.bias_downconv
        self.bias_deconv = args.bias_deconv
        self.bias_D_conv = args.bias_D_conv
        # clip D values
        self.d_clip_weights = False
        # apply VBN or regular BN?
        self.disable_vbn = False
        self.save_path = args.save_path
        # num of updates to be applied to D before G
        # this is k in original GAN paper (https://arxiv.org/abs/1406.2661)
        self.disc_updates = 1
        # set preemph factor
        self.preemph = args.preemph
        if self.preemph > 0:
            print('*** Applying pre-emphasis of {} ***'.format(self.preemph))
        else:
            print('--- No pre-emphasis applied ---')
        # canvas size
        self.canvas_size = args.canvas_size
        self.deactivated_noise = False
        # dilation factors per layer (only in atrous conv G config)
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # num fmaps for AutoEncoder SEGAN (v1)
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        # Define D fmaps
        self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.init_noise_std = args.init_noise_std
        self.disc_noise_std = tf.Variable(self.init_noise_std, trainable=False)
        self.disc_noise_std_summ = scalar_summary('disc_noise_std',
                                                  self.disc_noise_std)
        self.e2e_dataset = args.e2e_dataset
        # G's supervised loss weight
        self.l1_weight = args.init_l1_weight
        self.l1_lambda = tf.Variable(self.l1_weight, trainable=False)
        self.deactivated_l1 = False
        # define the functions
        self.discriminator = discriminator
        # register G non linearity
        self.g_nl = args.g_nl
        if args.g_type == 'ae':
            self.generator = AEGenerator(self)
        elif args.g_type == 'dwave':
            self.generator = Generator(self)
        else:
            raise ValueError('Unrecognized G type {}'.format(args.g_type))
        self.build_model(args)

    def build_model(self, config):
        all_d_grads = []
        all_g_grads = []
        d_opt = tf.train.RMSPropOptimizer(config.d_learning_rate)
        g_opt = tf.train.RMSPropOptimizer(config.g_learning_rate)
        #d_opt = tf.train.AdamOptimizer(config.d_learning_rate,
        #                               beta1=config.beta_1)
        #g_opt = tf.train.AdamOptimizer(config.g_learning_rate,
        #                               beta1=config.beta_1)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for idx, device in enumerate(self.devices):
                with tf.device("/%s" % device):
                    with tf.name_scope("device_%s" % idx):
                        with variables_on_gpu0():
                            self.build_model_single_gpu(idx)
                            d_grads = d_opt.compute_gradients(self.d_losses[-1],
                                                              var_list=self.d_vars)
                            g_grads = g_opt.compute_gradients(self.g_losses[-1],
                                                              var_list=self.g_vars)
                            all_d_grads.append(d_grads)
                            all_g_grads.append(g_grads)

        avg_d_grads = average_gradients(all_d_grads)
        avg_g_grads = average_gradients(all_g_grads)
        self.d_opt = d_opt.apply_gradients(avg_d_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)


    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # create the nodes to load for input pipeline
            filename_queue = tf.train.string_input_producer(string_tensor=[self.e2e_dataset], capacity=128)
            self.get_wav, self.get_noisy, self.get_noisy_only = read_and_decode(filename_queue,
                                                           self.canvas_size,
                                                           self.preemph)
        # load the data to input pipeline
        wavbatch, \
        noisybatch, \
        noisyonlybatch = tf.train.shuffle_batch([self.get_wav,
                                             self.get_noisy,
                                             self.get_noisy_only],
                                             batch_size=self.batch_size,
                                             num_threads=128,
                                             capacity=1000 + 3 * self.batch_size,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy_and_noisy_only')
        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []
            self.gtruth_noisy_only = []

        self.gtruth_wavs.append(wavbatch)
        self.gtruth_noisy.append(noisybatch)

        # add channels dimension to manipulate in D and G
        wavbatch = tf.expand_dims(wavbatch, -1)
        noisybatch = tf.expand_dims(noisybatch, -1)
    
        self.gtruth_noisy_only.append(noisyonlybatch)
        noisyonlybatch = tf.expand_dims(noisyonlybatch, -1)

        # by default leaky relu is used
        do_prelu = False
        if self.g_nl == 'prelu':
            do_prelu = True
        if gpu_idx == 0:
            #self.sample_wavs = tf.placeholder(tf.float32, [self.batch_size,
            #                                               self.canvas_size],
            #                                  name='sample_wavs')
            ref_Gs = self.generator(noisybatch, is_ref=True,
                                    spk=None,
                                    do_prelu=do_prelu)
            print('num of G returned: ', len(ref_Gs))
            
            self.reference_G = ref_Gs[0]
            self.ref_z = ref_Gs[1]
            # if do_prelu:
            #    self.ref_alpha = ref_Gs[2:]
            #    self.alpha_summ = []
            #    for m, ref_alpha in enumerate(self.ref_alpha):
            #        # add a summary per alpha
            #        self.alpha_summ.append(histogram_summary('alpha_{}'.format(m),
            #                                                 ref_alpha))
            # make a dummy copy of discriminator to have variables and then
            # be able to set up the variable reuse for all other devices
            # merge along channels and this would be a real batch
            dummy_joint = tf.concat([wavbatch, noisybatch], 2)
            dummy = discriminator(self, dummy_joint,
                                  reuse=False)

        G1, G2, H1, H2, z1, z2 = self.generator(noisybatch, is_ref=False, spk=None,
                                                do_prelu=do_prelu)

        self.Gs.append(G1)
        self.zs.append(z1)
        
        # add new dimension to merge with other pairs 
        D_rl_joint = tf.concat([wavbatch, noisybatch], 2)
        D_fk_joint = tf.concat([G1, noisybatch], 2)
        
        #for noise condition
        D_rl_joint_noise = tf.concat([noisyonlybatch, noisybatch], 2)

        # build rl discriminator
        dropout_enable = True if self.decoder_type=='II' else False
        d_rl_logits = discriminator(self, D_rl_joint, dropout_enable=dropout_enable, reuse=True)
        # build fk G discriminator
        d_fk_logits = discriminator(self, D_fk_joint, dropout_enable=dropout_enable, reuse=True)
        # make disc variables summaries
        self.d_rl_sum = histogram_summary("d_real", d_rl_logits)
        self.d_fk_sum = histogram_summary("d_fake", d_fk_logits)
        self.gen_audio_summ = audio_summary('G_audio', G1)
        self.gen_summ = histogram_summary('G_wav', G1)
        
        #self.d_nfk_sum = histogram_summary("d_noisyfake", d_nfk_logits)

        self.rl_audio_summ = audio_summary('real_audio', wavbatch)
        self.real_w_summ = histogram_summary('real_wav', wavbatch)
        self.noisy_audio_summ = audio_summary('noisy_audio', noisybatch)
        self.noisy_w_summ = histogram_summary('noisy_wav', noisybatch)
        self.noisy_audio_noise_summ = audio_summary('noisy_audio_only', noisyonlybatch)
        self.noisy_w_only_summ = histogram_summary('noisy_wav_only', noisyonlybatch)

        if gpu_idx == 0:
            self.g_losses = []
            self.g_l1_losses = []
            self.g_l1_losses_only = []
            self.g_adv_losses = []
            self.g_adv_losses_only = []
            self.d_rl_losses = []
            self.d_rl_losses_only = []
            self.d_fk_losses = []
            self.d_fk_losses_only = []
            #self.d_nfk_losses = []
            self.d_losses = []
            self.hinge_losses = []

        if self.decoder_type == 'II':
            d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.-self.d_label_smooth))
        else:
            d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
        d_fk_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 0.))
        g_adv_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 1.))
        # Add the L1 loss to G
        g_l1_loss = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(G1,
                                                                  wavbatch)))
        self.g_l1_losses.append(g_l1_loss)
        self.g_adv_losses.append(g_adv_loss)
        self.d_rl_losses.append(d_rl_loss)
        self.d_fk_losses.append(d_fk_loss)

        #d_nfk_loss = tf.reduce_mean(tf.squared_difference(d_nfk_logits, 0.))

        #compute encoder laten varible distence
        g_loss = g_adv_loss + g_l1_loss
        d_loss = d_rl_loss + d_fk_loss

        self.g_losses.append(g_loss)
        #self.d_nfk_losses.append(d_nfk_loss)
        self.d_losses.append(d_loss)
        
        #self.d_nfk_loss_sum = scalar_summary("d_nfk_loss",
        #                                     d_nfk_loss)
        self.g_loss_sum = scalar_summary("g_loss", g_loss)
        self.d_loss_sum = scalar_summary("d_loss", d_loss)
        self.d_rl_loss_sum = scalar_summary("d_rl_loss", d_rl_loss)
        self.d_fk_loss_sum = scalar_summary("d_fk_loss", d_fk_loss)
        self.g_loss_l1_sum = scalar_summary("g_l1_loss", g_l1_loss)
        self.g_loss_adv_sum = scalar_summary("g_adv_loss", g_adv_loss)

        if gpu_idx == 0:
            self.get_vars()

    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars_dict = {}
        self.g_vars_dict = {}
        for var in t_vars:
            if var.name.startswith('d_'):
                self.d_vars_dict[var.name] = var
            if var.name.startswith('g_'):
                self.g_vars_dict[var.name] = var
        self.d_vars = list(self.d_vars_dict.values())
        self.g_vars = list(self.g_vars_dict.values())
        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in t_vars:
            assert x in self.g_vars or x in self.d_vars, x.name
        self.all_vars = t_vars
        if self.d_clip_weights:
            print('Clipping D weights')
            self.d_clip = [v.assign(tf.clip_by_value(v, -0.05, 0.05)) for v in self.d_vars]
        else:
            print('Not clipping D weights')

    def vbn(self, tensor, name):
        if self.disable_vbn:
            class Dummy(object):
                # Do nothing here, no bnorm
                def __init__(self, tensor, ignored):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def train(self, config, devices):
        """ Train the SEGAN """

        print('Initializing optimizers...')
        # init optimizers
        d_opt = self.d_opt
        g_opt = self.g_opt
        num_devices = len(devices)

        try:
            init = tf.global_variables_initializer()
        except AttributeError:
            # fall back to old implementation
            init = tf.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        g_summs = [self.d_fk_sum,
                   self.d_fk_loss_sum,
                   self.g_loss_sum,
                   self.g_loss_l1_sum,
                   self.g_loss_adv_sum,
                   self.gen_summ,
                   self.gen_audio_summ]
        d_sums = [self.d_loss_sum,
                  self.d_rl_sum,
                  self.d_rl_loss_sum,
                  self.rl_audio_summ,
                  self.real_w_summ,
                  self.disc_noise_std_summ]

        # if we have prelus, add them to summary
        if hasattr(self, 'alpha_summ'):
            g_summs += self.alpha_summ
        self.g_sum = tf.summary.merge(g_summs)
        self.d_sum = tf.summary.merge(d_sums)

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'), self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        # Hang onto a copy of wavs so we can feed the same one every time
        # we store samples to disk for hearing
        # pick a single batch
        sample_noisy, sample_wav, sample_noise_only, \
        sample_z = self.sess.run([self.gtruth_noisy[0],
                                  self.gtruth_wavs[0],
                                  self.gtruth_noisy_only[0],
                                  self.zs[0]])
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        print('sample noisy_only shape: ', sample_noise_only.shape)
        print('sample z shape: ', sample_z.shape)

        save_path = config.save_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        d_fk_losses = []
        d_fk_only_losses = []
        #d_nfk_losses = []
        d_rl_losses = []
        d_rl_only_losses = []
        hinge_losses = []
        g_adv_only_losses = []
        g_adv_losses = []
        g_l1_only_losses = []
        g_l1_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()

                d_list = [d_opt, self.d_fk_losses[0], self.d_rl_losses[0]]
                g_list = [g_opt, self.g_adv_losses[0], self.g_l1_losses[0]]

                if counter % config.save_freq == 0:
                    for d_iter in range(self.disc_updates):
                        d_out = self.sess.run(d_list+[self.d_sum])
                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)
                        #d_nfk_loss, \

                    # now G iterations
                    g_out = self.sess.run(g_list+[self.g_sum])
                    _d_sum = d_out.pop(-1)
                    _g_sum = g_out.pop(-1)

                else:
                    for d_iter in range(self.disc_updates):
                        d_out = self.sess.run(d_list)

                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)
                        #d_nfk_loss, \

                    # now G iterations
                    g_out = self.sess.run(g_list)

                end = timeit.default_timer()
                batch_timings.append(end - start)

                _, d_fk_loss, d_rl_loss = d_out
                _, g_adv_loss, g_l1_loss = g_out
                d_fk_losses.append(d_fk_loss)
                d_rl_losses.append(d_rl_loss)
                g_adv_losses.append(g_adv_loss)
                g_l1_losses.append(g_l1_loss)

                print('{}/{} (epoch {}), d_rl_loss = {:.5f}, '
                      'd_fk_loss = {:.5f}, '#d_nfk_loss = {:.5f}, '
                      'g_adv_loss = {:.5f}, g_l1_loss = {:.5f},'
                      'time/batch = {:.5f},'
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    d_rl_loss,
                                                    d_fk_loss,
                                                    #d_nfk_loss,
                                                    g_adv_loss,
                                                    g_l1_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter / num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)

                    fdict = {self.gtruth_noisy[0]:sample_noisy,
                             self.zs[0]:sample_z}
                    canvas_w = self.sess.run(self.Gs[0],
                                             feed_dict=fdict)

                    swaves = sample_wav
                    sample_dif = sample_wav - sample_noisy
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m,
                                                           np.max(canvas_w[m]),
                                                           np.min(canvas_w[m])))
                        wavfile.write(os.path.join(save_path,
                                                   'sample_{}-'
                                                   '{}.wav'.format(counter, m)),
                                      int(16e3),
                                      de_emph(canvas_w[m],
                                              self.preemph))
                        m_gtruth_path = os.path.join(save_path, 'gtruth_{}.'
                                                                'wav'.format(m))
                        if not os.path.exists(m_gtruth_path):
                            wavfile.write(os.path.join(save_path,
                                                       'gtruth_{}.'
                                                       'wav'.format(m)),
                                                       int(16e3),
                                                       de_emph(swaves[m],
                                                       self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'noisy_{}.'
                                                       'wav'.format(m)),
                                                       int(16e3),
                                                       de_emph(sample_noisy[m],
                                                       self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'dif_{}.wav'.format(m)),
                                                       int(16e3),
                                                       de_emph(sample_dif[m],
                                                       self.preemph))
                        np.savetxt(os.path.join(save_path, 'd_rl_losses.txt'),
                                   d_rl_losses)
                        np.savetxt(os.path.join(save_path, 'd_fk_losses.txt'),
                                   d_fk_losses)
                        np.savetxt(os.path.join(save_path, 'g_adv_losses.txt'),
                                   g_adv_losses)
                        np.savetxt(os.path.join(save_path, 'g_l1_losses.txt'),
                                   g_l1_losses)

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    # re-set batch idx
                    batch_idx = 0
                    # check if we have to deactivate L1
                    if curr_epoch >= config.l1_remove_epoch and self.deactivated_l1 == False:
                        print('** Deactivating L1 factor! **')
                        self.sess.run(tf.assign(self.l1_lambda, 0.))
                        self.deactivated_l1 = True
                    # check if we have to start decaying noise (if any)
                    if curr_epoch >= config.denoise_epoch and self.deactivated_noise == False:
                        # apply noise std decay rate
                        decay = config.noise_decay
                        if not hasattr(self, 'curr_noise_std'):
                            self.curr_noise_std = self.init_noise_std
                        new_noise_std = decay * self.curr_noise_std
                        if new_noise_std < config.denoise_lbound:
                            print('New noise std {} < lbound {}, setting 0.'.format(new_noise_std, config.denoise_lbound))
                            print('** De-activating noise layer **')
                            # it it's lower than a lower bound, cancel out completely
                            new_noise_std = 0.
                            self.deactivated_noise = True
                        else:
                            print('Applying decay {} to noise std {}: {}'.format(decay, self.curr_noise_std, new_noise_std))
                        self.sess.run(tf.assign(self.disc_noise_std, new_noise_std))
                        self.curr_noise_std = new_noise_std
                if curr_epoch >= config.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)

    def clean(self, x):
        """ clean a utterance x
            x: numpy array containing the normalized noisy waveform
        """
        c_res = None
        for beg_i in range(0, x.shape[0], self.canvas_size):
            if x.shape[0] - beg_i  < self.canvas_size:
                length = x.shape[0] - beg_i
                pad = (self.canvas_size) - length
            else:
                length = self.canvas_size
                pad = 0
            x_ = np.zeros((self.batch_size, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]
            print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
            fdict = {self.gtruth_noisy[0]:x_}
            canvas_w = self.sess.run(self.Gs[0],
                                     feed_dict=fdict)[0]
            canvas_w = canvas_w.reshape((self.canvas_size))
            print('canvas w shape: ', canvas_w.shape)
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                # get rid of last padded samples
                canvas_w = canvas_w[:-pad]
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        # deemphasize
        c_res = de_emph(c_res, self.preemph)
        return c_res    


class SEGAN_III(SEGAN_I):
    def __init__(self, sess, args, devices, infer=False, name='SEGAN_III'):
        super(SEGAN_III, self).__init__(
            sess=sess,
            args=args,
            devices=devices,
            infer=infer,
            name=name)

    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # create the nodes to load for input pipeline
            filename_queue = tf.train.string_input_producer(string_tensor=[self.e2e_dataset], capacity=128)
            self.get_wav, self.get_noisy, self.get_noisy_only = read_and_decode(filename_queue,
                                                           self.canvas_size,
                                                           self.preemph)
        # load the data to input pipeline
        wavbatch, \
        noisybatch, \
        noisyonlybatch = tf.train.shuffle_batch([self.get_wav,
                                             self.get_noisy,
                                             self.get_noisy_only],
                                             batch_size=self.batch_size,
                                             num_threads=128,
                                             capacity=1000 + 3 * self.batch_size,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy_and_noisy_only')
        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []
            self.gtruth_noisy_only = []

        self.gtruth_wavs.append(wavbatch)
        self.gtruth_noisy.append(noisybatch)

        # add channels dimension to manipulate in D and G
        wavbatch = tf.expand_dims(wavbatch, -1)
        noisybatch = tf.expand_dims(noisybatch, -1)
    
        self.gtruth_noisy_only.append(noisyonlybatch)
        noisyonlybatch = tf.expand_dims(noisyonlybatch, -1)

        # by default leaky relu is used
        do_prelu = False
        if self.g_nl == 'prelu':
            do_prelu = True
        if gpu_idx == 0:
            #self.sample_wavs = tf.placeholder(tf.float32, [self.batch_size,
            #                                               self.canvas_size],
            #                                  name='sample_wavs')
            ref_Gs = self.generator(noisybatch, is_ref=True,
                                    spk=None,
                                    do_prelu=do_prelu)
            print('num of G returned: ', len(ref_Gs))
            self.reference_G2 = ref_Gs[1]
            self.ref_z = []
            # if do_prelu:
            #    self.ref_alpha = ref_Gs[2:]
            #    self.alpha_summ = []
            #    for m, ref_alpha in enumerate(self.ref_alpha):
            #        # add a summary per alpha
            #        self.alpha_summ.append(histogram_summary('alpha_{}'.format(m),
            #                                                 ref_alpha))
            # make a dummy copy of discriminator to have variables and then
            # be able to set up the variable reuse for all other devices
            # merge along channels and this would be a real batch
            dummy_joint = tf.concat([wavbatch, noisybatch], 2)
            dummy = discriminator(self, dummy_joint,
                                  reuse=False)

        G1, G2, H1, H2, z1, z2 = self.generator(noisybatch, is_ref=False, spk=None,
                                                do_prelu=do_prelu)

        self.Gs.append(G2)
        self.zs.append(z2)
        D_fk_joint_noise = tf.concat([G2, noisybatch], 2)
        
        # add new dimension to merge with other pairs 
        D_rl_joint = tf.concat([wavbatch, noisybatch], 2)

        #for noise condition
        D_rl_joint_noise = tf.concat([noisyonlybatch, noisybatch], 2)
        
        # discriminator for noise condition
        d_rl_logits_noise = discriminator(self, D_rl_joint_noise, dropout_enable=True, reuse=True)
        d_fk_logits_noise = discriminator(self, D_fk_joint_noise, dropout_enable=True, reuse=True)
        # make disc variables summaries
        self.d_fk_noise_sum = histogram_summary("d_fake_noise", d_fk_logits_noise)
        self.d_rl_noise_sum = histogram_summary("d_real_noise", d_rl_logits_noise)
        self.gen_audio_noise_summ = audio_summary('G_audio_only', G2)
        self.gen_only_summ = histogram_summary('G_wav_only', G2)

        #self.d_nfk_sum = histogram_summary("d_noisyfake", d_nfk_logits)

        self.rl_audio_summ = audio_summary('real_audio', wavbatch)
        self.real_w_summ = histogram_summary('real_wav', wavbatch)
        self.noisy_audio_summ = audio_summary('noisy_audio', noisybatch)
        self.noisy_w_summ = histogram_summary('noisy_wav', noisybatch)
        self.noisy_audio_noise_summ = audio_summary('noisy_audio_only', noisyonlybatch)
        self.noisy_w_only_summ = histogram_summary('noisy_wav_only', noisyonlybatch)

        if gpu_idx == 0:
            self.g_losses = []
            self.g_l1_losses = []
            self.g_l1_losses_only = []
            self.g_adv_losses = []
            self.g_adv_losses_only = []
            self.d_rl_losses = []
            self.d_rl_losses_only = []
            self.d_fk_losses = []
            self.d_fk_losses_only = []
            #self.d_nfk_losses = []
            self.d_losses = []
            self.hinge_losses = []

        d_rl_loss_only = tf.reduce_mean(tf.squared_difference(d_rl_logits_noise, 1.-self.d_label_smooth))
        d_fk_loss_only = tf.reduce_mean(tf.squared_difference(d_fk_logits_noise, 0.))
        g_adv_loss_only = tf.reduce_mean(tf.squared_difference(d_fk_logits_noise, 1.))
        # Add the L1 loss to G
        g_l1_loss_only = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(G2,
                                                                  noisyonlybatch)))
        self.g_l1_losses_only.append(g_l1_loss_only)
        self.g_adv_losses_only.append(g_adv_loss_only)
        self.d_rl_losses_only.append(d_rl_loss_only)
        self.d_fk_losses_only.append(d_fk_loss_only)

        self.d_rl_loss_only_sum = scalar_summary("d_rl_loss_only", d_rl_loss_only)
        self.d_fk_loss_only_sum = scalar_summary("d_fk_loss_only", d_fk_loss_only)
        self.g_loss_l1_only_sum = scalar_summary("g_l1_loss_only", g_l1_loss_only)
        self.g_loss_adv_only_sum = scalar_summary("g_adv_loss_only", g_adv_loss_only)

        #d_nfk_loss = tf.reduce_mean(tf.squared_difference(d_nfk_logits, 0.))
        d_loss = d_rl_loss_only + d_fk_loss_only

        #compute encoder laten varible distence
        g_loss = g_adv_loss_only + g_l1_loss_only

        self.g_losses.append(g_loss)
        #self.d_nfk_losses.append(d_nfk_loss)
        self.d_losses.append(d_loss)
        
        #self.d_nfk_loss_sum = scalar_summary("d_nfk_loss",
        #                                     d_nfk_loss)
        self.g_loss_sum = scalar_summary("g_loss", g_loss)
        self.d_loss_sum = scalar_summary("d_loss", d_loss)

        if gpu_idx == 0:
            self.get_vars()

    def train(self, config, devices):
        """ Train the SEGAN """

        print('Initializing optimizers...')
        # init optimizers
        d_opt = self.d_opt
        g_opt = self.g_opt
        num_devices = len(devices)

        try:
            init = tf.global_variables_initializer()
        except AttributeError:
            # fall back to old implementation
            init = tf.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        g_summs = [#self.d_nfk_sum,
                   #self.d_nfk_loss_sum,
                   self.d_fk_noise_sum,
                   self.d_fk_loss_only_sum,
                   self.g_loss_sum,
                   self.g_loss_l1_only_sum,
                   self.g_loss_adv_only_sum,
                   self.gen_only_summ,
                   self.gen_audio_noise_summ]
        d_sums = [self.d_loss_sum,
                  self.d_rl_noise_sum,
                  self.d_rl_loss_only_sum,
                  self.rl_audio_summ,
                  self.real_w_summ,
                  self.disc_noise_std_summ]

        # if we have prelus, add them to summary
        if hasattr(self, 'alpha_summ'):
            g_summs += self.alpha_summ
        self.g_sum = tf.summary.merge(g_summs)
        self.d_sum = tf.summary.merge(d_sums)

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'), self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        # Hang onto a copy of wavs so we can feed the same one every time
        # we store samples to disk for hearing
        # pick a single batch

        sample_noisy, sample_wav, sample_noise_only, \
        sample_z = self.sess.run([self.gtruth_noisy[0],
                                  self.gtruth_wavs[0],
                                  self.gtruth_noisy_only[0],
                                  self.zs[0]])
        print('sample z shape: ', sample_z.shape)
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        print('sample noisy_only shape: ', sample_noise_only.shape)

        save_path = config.save_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        d_fk_losses = []
        d_fk_only_losses = []
        #d_nfk_losses = []
        d_rl_losses = []
        d_rl_only_losses = []
        hinge_losses = []
        g_adv_only_losses = []
        g_adv_losses = []
        g_l1_only_losses = []
        g_l1_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()

                d_list = [d_opt, self.d_fk_losses_only[0], self.d_rl_losses_only[0]]
                g_list = [g_opt, self.g_adv_losses_only[0], self.g_l1_losses_only[0]]
                
                if counter % config.save_freq == 0:
                    for d_iter in range(self.disc_updates):
                        d_out = self.sess.run(d_list+[self.d_sum])
                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)
                        #d_nfk_loss, \

                    # now G iterations
                    g_out = self.sess.run(g_list+[self.g_sum])
                    _d_sum = d_out.pop(-1)
                    _g_sum = g_out.pop(-1)

                else:
                    for d_iter in range(self.disc_updates):
                        d_out = self.sess.run(d_list)

                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)
                        #d_nfk_loss, \

                    # now G iterations
                    g_out = self.sess.run(g_list)

                end = timeit.default_timer()
                batch_timings.append(end - start)

                _, d_fk_only_loss, d_rl_only_loss = d_out
                _, g_adv_only_loss, g_l1_only_loss = g_out
                d_fk_only_losses.append(d_fk_only_loss)
                d_rl_only_losses.append(d_rl_only_loss)
                g_adv_only_losses.append(g_adv_only_loss)
                g_l1_only_losses.append(g_l1_only_loss)

                print('{}/{} (epoch {}), d_rl_only_loss = {:.5f}, '
                      'd_fk_only_loss = {:.5f}, '
                      'g_adv_only_loss = {:.5f}, g_l1_only_loss = {:.5f},'
                      'time/batch = {:.5f},'
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    d_rl_only_loss,
                                                    d_fk_only_loss,
                                                    #d_nfk_loss,
                                                    g_adv_only_loss,
                                                    g_l1_only_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter / num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)

                    fdict = {self.gtruth_noisy[0]:sample_noisy,
                             self.zs[0]:sample_z}
                    
                    canvas_w = self.sess.run(self.Gs2[0],
                                             feed_dict=fdict)
                    swaves = sample_wav
                    sample_dif = sample_wav - sample_noisy
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m,
                                                           np.max(canvas_w[m]),
                                                           np.min(canvas_w[m])))
                        wavfile.write(os.path.join(save_path,
                                                   'sample_{}-'
                                                   '{}.wav'.format(counter, m)),
                                      int(16e3),
                                      de_emph(canvas_w[m],
                                              self.preemph))
                        m_gtruth_path = os.path.join(save_path, 'gtruth_{}.'
                                                                'wav'.format(m))
                        if not os.path.exists(m_gtruth_path):
                            wavfile.write(os.path.join(save_path,
                                                       'gtruth_{}.'
                                                       'wav'.format(m)),
                                          int(16e3),
                                          de_emph(swaves[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'noisy_{}.'
                                                       'wav'.format(m)),
                                          int(16e3),
                                          de_emph(sample_noisy[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'dif_{}.wav'.format(m)),
                                          int(16e3),
                                          de_emph(sample_dif[m],
                                                  self.preemph))
                        np.savetxt(os.path.join(save_path, 'd_rl_losses.txt'),
                                   d_rl_only_losses)
                        np.savetxt(os.path.join(save_path, 'd_fk_losses.txt'),
                                   d_fk_only_losses)
                        np.savetxt(os.path.join(save_path, 'g_adv_losses.txt'),
                                   g_adv_only_losses)
                        np.savetxt(os.path.join(save_path, 'g_l1_losses.txt'),
                                   g_l1_only_losses)

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    # re-set batch idx
                    batch_idx = 0
                    # check if we have to deactivate L1
                    if curr_epoch >= config.l1_remove_epoch and self.deactivated_l1 == False:
                        print('** Deactivating L1 factor! **')
                        self.sess.run(tf.assign(self.l1_lambda, 0.))
                        self.deactivated_l1 = True
                    # check if we have to start decaying noise (if any)
                    if curr_epoch >= config.denoise_epoch and self.deactivated_noise == False:
                        # apply noise std decay rate
                        decay = config.noise_decay
                        if not hasattr(self, 'curr_noise_std'):
                            self.curr_noise_std = self.init_noise_std
                        new_noise_std = decay * self.curr_noise_std
                        if new_noise_std < config.denoise_lbound:
                            print('New noise std {} < lbound {}, setting 0.'.format(new_noise_std, config.denoise_lbound))
                            print('** De-activating noise layer **')
                            # it it's lower than a lower bound, cancel out completely
                            new_noise_std = 0.
                            self.deactivated_noise = True
                        else:
                            print('Applying decay {} to noise std {}: {}'.format(decay, self.curr_noise_std, new_noise_std))
                        self.sess.run(tf.assign(self.disc_noise_std, new_noise_std))
                        self.curr_noise_std = new_noise_std
                if curr_epoch >= config.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)
  

class SEGAN_IV(SEGAN_I):
    def __init__(self, sess, args, devices, infer=False, name='SEGAN_IV'):
        super(SEGAN_IV, self).__init__(
            sess=sess,
            args=args,
            devices=devices,
            infer=infer,
            name=name)
        
    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # create the nodes to load for input pipeline
            filename_queue = tf.train.string_input_producer(string_tensor=[self.e2e_dataset], capacity=128)
            self.get_wav, self.get_noisy, self.get_noisy_only = read_and_decode(filename_queue,
                                                           self.canvas_size,
                                                           self.preemph)
        # load the data to input pipeline
        wavbatch, \
        noisybatch, \
        noisyonlybatch = tf.train.shuffle_batch([self.get_wav,
                                             self.get_noisy,
                                             self.get_noisy_only],
                                             batch_size=self.batch_size,
                                             num_threads=128,
                                             capacity=1000 + 3 * self.batch_size,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy_and_noisy_only')
        if gpu_idx == 0:
            self.Gs1, self.Gs2 = [], []
            self.zs1, self.zs2 = [], []
            self.gtruth_wavs = []
            self.gtruth_noisy = []
            self.gtruth_noisy_only = []

        self.gtruth_wavs.append(wavbatch)
        self.gtruth_noisy.append(noisybatch)

        # add channels dimension to manipulate in D and G
        wavbatch = tf.expand_dims(wavbatch, -1)
        noisybatch = tf.expand_dims(noisybatch, -1)
    
        self.gtruth_noisy_only.append(noisyonlybatch)
        noisyonlybatch = tf.expand_dims(noisyonlybatch, -1)

        # by default leaky relu is used
        do_prelu = False
        if self.g_nl == 'prelu':
            do_prelu = True
        if gpu_idx == 0:
            #self.sample_wavs = tf.placeholder(tf.float32, [self.batch_size,
            #                                               self.canvas_size],
            #                                  name='sample_wavs')
            ref_Gs = self.generator(noisybatch, is_ref=True,
                                    spk=None,
                                    do_prelu=do_prelu)
            print('num of G returned: ', len(ref_Gs))
            self.reference_G1 = ref_Gs[0]
            self.reference_G2 = ref_Gs[1]
            self.ref_z = []
            # if do_prelu:
            #    self.ref_alpha = ref_Gs[2:]
            #    self.alpha_summ = []
            #    for m, ref_alpha in enumerate(self.ref_alpha):
            #        # add a summary per alpha
            #        self.alpha_summ.append(histogram_summary('alpha_{}'.format(m),
            #                                                 ref_alpha))
            # make a dummy copy of discriminator to have variables and then
            # be able to set up the variable reuse for all other devices
            # merge along channels and this would be a real batch
            dummy_joint = tf.concat([wavbatch, noisybatch], 2)
            dummy = discriminator(self, dummy_joint,
                                  reuse=False)

        G1, G2, H1, H2, z1, z2 = self.generator(noisybatch, is_ref=False, spk=None,
                                                do_prelu=do_prelu)

        self.Gs1.append(G1)
        self.zs1.append(z1)
        D_fk_joint = tf.concat([G1, noisybatch], 2)
        self.Gs2.append(G2)
        self.zs2.append(z2)
        D_fk_joint_noise = tf.concat([G2, noisybatch], 2)
        
        # add new dimension to merge with other pairs 
        D_rl_joint = tf.concat([wavbatch, noisybatch], 2)

        #for noise condition
        D_rl_joint_noise = tf.concat([noisyonlybatch, noisybatch], 2)

        # build rl discriminator
        d_rl_logits = discriminator(self, D_rl_joint, reuse=True)
        # build fk G discriminator
        d_fk_logits = discriminator(self, D_fk_joint, reuse=True)
        # make disc variables summaries
        self.d_rl_sum = histogram_summary("d_real", d_rl_logits)
        self.d_fk_sum = histogram_summary("d_fake", d_fk_logits)
        self.gen_audio_summ = audio_summary('G_audio', G1)
        self.gen_summ = histogram_summary('G_wav', G1)
        
        # discriminator for noise condition
        d_rl_logits_noise = discriminator(self, D_rl_joint_noise, dropout_enable=True, reuse=True)
        d_fk_logits_noise = discriminator(self, D_fk_joint_noise, dropout_enable=True, reuse=True)
        # d_rl_logits_noise = discriminator(self, D_rl_joint_noise, reuse=True)
        # d_fk_logits_noise = discriminator(self, D_fk_joint_noise, reuse=True)

        # make disc variables summaries
        self.d_fk_noise_sum = histogram_summary("d_fake_noise", d_fk_logits_noise)
        self.d_rl_noise_sum = histogram_summary("d_real_noise", d_rl_logits_noise)
        self.gen_audio_noise_summ = audio_summary('G_audio_only', G2)
        self.gen_only_summ = histogram_summary('G_wav_only', G2)

        #self.d_nfk_sum = histogram_summary("d_noisyfake", d_nfk_logits)

        self.rl_audio_summ = audio_summary('real_audio', wavbatch)
        self.real_w_summ = histogram_summary('real_wav', wavbatch)
        self.noisy_audio_summ = audio_summary('noisy_audio', noisybatch)
        self.noisy_w_summ = histogram_summary('noisy_wav', noisybatch)
        self.noisy_audio_noise_summ = audio_summary('noisy_audio_only', noisyonlybatch)
        self.noisy_w_only_summ = histogram_summary('noisy_wav_only', noisyonlybatch)

        if gpu_idx == 0:
            self.g_losses = []
            self.g_l1_losses = []
            self.g_l1_losses_only = []
            self.g_adv_losses = []
            self.g_adv_losses_only = []
            self.d_rl_losses = []
            self.d_rl_losses_only = []
            self.d_fk_losses = []
            self.d_fk_losses_only = []
            #self.d_nfk_losses = []
            self.d_losses = []
            self.hinge_losses = []

        d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
        d_fk_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 0.))
        g_adv_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 1.))
        # Add the L1 loss to G
        g_l1_loss = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(G1,
                                                                  wavbatch)))
        self.g_l1_losses.append(g_l1_loss)
        self.g_adv_losses.append(g_adv_loss)
        self.d_rl_losses.append(d_rl_loss)
        self.d_fk_losses.append(d_fk_loss)

        self.d_rl_loss_sum = scalar_summary("d_rl_loss", d_rl_loss)
        self.d_fk_loss_sum = scalar_summary("d_fk_loss", d_fk_loss)
        self.g_loss_l1_sum = scalar_summary("g_l1_loss", g_l1_loss)
        self.g_loss_adv_sum = scalar_summary("g_adv_loss", g_adv_loss)

        d_rl_loss_only = tf.reduce_mean(tf.squared_difference(d_rl_logits_noise, 1.-self.d_label_smooth))
        # d_rl_loss_only = tf.reduce_mean(tf.squared_difference(d_rl_logits_noise, 1.))
        d_fk_loss_only = tf.reduce_mean(tf.squared_difference(d_fk_logits_noise, 0.))
        g_adv_loss_only = tf.reduce_mean(tf.squared_difference(d_fk_logits_noise, 1.))
        # Add the L1 loss to G
        g_l1_loss_only = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(G2,
                                                                  noisyonlybatch)))
        self.g_l1_losses_only.append(g_l1_loss_only)
        self.g_adv_losses_only.append(g_adv_loss_only)
        self.d_rl_losses_only.append(d_rl_loss_only)
        self.d_fk_losses_only.append(d_fk_loss_only)

        self.d_rl_loss_only_sum = scalar_summary("d_rl_loss_only", d_rl_loss_only)
        self.d_fk_loss_only_sum = scalar_summary("d_fk_loss_only", d_fk_loss_only)
        self.g_loss_l1_only_sum = scalar_summary("g_l1_loss_only", g_l1_loss_only)
        self.g_loss_adv_only_sum = scalar_summary("g_adv_loss_only", g_adv_loss_only)

        #d_nfk_loss = tf.reduce_mean(tf.squared_difference(d_nfk_logits, 0.))

        d_loss = d_rl_loss + d_fk_loss + d_rl_loss_only + d_fk_loss_only

        #compute encoder laten varible distence
        # H1, H2 = tf.nn.l2_normalize(H1, axis=1), tf.nn.l2_normalize(H2, axis=1)
        d = tf.sqrt(tf.reduce_sum(tf.pow(H1-H2, 2), 1, keepdims=True))
        margin = 10.0
        hinge_loss = tf.reduce_mean(tf.square(tf.maximum((margin - d),0)))
        
        if self.decoder_type == 'V':
            g_loss = g_adv_loss + g_l1_loss + g_adv_loss_only + g_l1_loss_only + hinge_loss
        else:
            g_loss = g_adv_loss + g_l1_loss + g_adv_loss_only + g_l1_loss_only
        self.hinge_losses.append(hinge_loss)
        self.hinge_loss_sum = scalar_summary("hinge_loss", tf.squeeze(hinge_loss))

        self.g_losses.append(g_loss)
        #self.d_nfk_losses.append(d_nfk_loss)
        self.d_losses.append(d_loss)
        
        #self.d_nfk_loss_sum = scalar_summary("d_nfk_loss",
        #                                     d_nfk_loss)
        self.g_loss_sum = scalar_summary("g_loss", g_loss)
        self.d_loss_sum = scalar_summary("d_loss", d_loss)

        if gpu_idx == 0:
            self.get_vars()

    def train(self, config, devices):
        """ Train the SEGAN """

        print('Initializing optimizers...')
        # init optimizers
        d_opt = self.d_opt
        g_opt = self.g_opt
        num_devices = len(devices)

        try:
            init = tf.global_variables_initializer()
        except AttributeError:
            # fall back to old implementation
            init = tf.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        g_summs = [self.d_fk_sum,
                   self.d_fk_loss_sum,
                   self.g_loss_l1_sum,
                   self.g_loss_adv_sum,
                   self.d_fk_noise_sum,
                   self.d_fk_loss_only_sum,
                   self.g_loss_l1_only_sum,
                   self.g_loss_adv_only_sum,
                   self.hinge_loss_sum,
                   self.gen_summ,
                   self.gen_audio_summ,
                   self.gen_only_summ,
                   self.gen_audio_noise_summ,
                   self.g_loss_sum]
        d_sums = [self.d_rl_noise_sum,
                  self.d_rl_loss_only_sum,
                  self.d_rl_sum,
                  self.d_rl_loss_sum,
                  self.d_loss_sum,
                  self.rl_audio_summ,
                  self.real_w_summ,
                  self.disc_noise_std_summ]

        # if we have prelus, add them to summary
        if hasattr(self, 'alpha_summ'):
            g_summs += self.alpha_summ
        self.g_sum = tf.summary.merge(g_summs)
        self.d_sum = tf.summary.merge(d_sums)

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'), self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        # Hang onto a copy of wavs so we can feed the same one every time
        # we store samples to disk for hearing
        # pick a single batch
        sample_noisy, sample_wav, sample_noise_only, sample_z1, \
        sample_z2 = self.sess.run([self.gtruth_noisy[0],
                                  self.gtruth_wavs[0],
                                  self.gtruth_noisy_only[0],
                                  self.zs1[0],
                                  self.zs2[0]])
        print('sample z1 shape: ', sample_z1.shape)
        print('sample z2 shape: ', sample_z2.shape)
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        print('sample noisy_only shape: ', sample_noise_only.shape)

        save_path = config.save_path
        counter = 0
        # count number of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        # last samples (those not filling a complete batch) are discarded
        num_batches = num_examples / self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        d_fk_losses = []
        d_fk_only_losses = []
        #d_nfk_losses = []
        d_rl_losses = []
        d_rl_only_losses = []
        hinge_losses = []
        g_adv_only_losses = []
        g_adv_losses = []
        g_l1_only_losses = []
        g_l1_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()

                d_list = [d_opt,
                          self.d_fk_losses_only[0],
                          self.d_rl_losses_only[0],
                          self.d_fk_losses[0],
                          #self.d_nfk_losses[0],
                          self.d_rl_losses[0]]
                g_list = [g_opt,
                          self.g_adv_losses_only[0],
                          self.g_l1_losses_only[0],
                          self.g_adv_losses[0],
                          self.g_l1_losses[0],
                          self.hinge_losses[0]]

                if counter % config.save_freq == 0:
                    for d_iter in range(self.disc_updates):
                        d_out = self.sess.run(d_list+[self.d_sum])
                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)
                        #d_nfk_loss, \

                    # now G iterations
                    g_out = self.sess.run(g_list+[self.g_sum])
                    _d_sum = d_out.pop(-1)
                    _g_sum = g_out.pop(-1)

                else:
                    for d_iter in range(self.disc_updates):
                        d_out = self.sess.run(d_list)

                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)
                        #d_nfk_loss, \

                    # now G iterations
                    g_out = self.sess.run(g_list)

                end = timeit.default_timer()
                batch_timings.append(end - start)

                _, d_fk_only_loss, d_rl_only_loss, d_fk_loss, d_rl_loss = d_out
                _, g_adv_only_loss, g_l1_only_loss, g_adv_loss, g_l1_loss, hinge_loss = g_out
                d_fk_losses.append(d_fk_loss)
                d_fk_only_losses.append(d_fk_only_loss)
                #d_nfk_losses.append(d_nfk_loss)
                d_rl_losses.append(d_rl_loss)
                d_rl_only_losses.append(d_rl_only_loss)
                g_adv_losses.append(g_adv_loss)
                g_adv_only_losses.append(g_adv_only_loss)
                g_l1_losses.append(g_l1_loss)
                g_l1_only_losses.append(g_l1_only_loss)
                hinge_losses.append(hinge_loss)

                print('{}/{} (epoch {}), d_rl_loss = {:.5f}, '
                      'd_fk_loss = {:.5f}, '#d_nfk_loss = {:.5f}, '
                      'd_rl_only_loss = {:.5f}, d_fk_only_loss = {:.5f},'
                      'g_adv_only_loss = {:.5f}, g_l1_only_loss = {:.5f},'
                      'g_adv_loss = {:.5f}, g_l1_loss = {:.5f},'
                      'hinge_loss = {:.5f},'
                      'time/batch = {:.5f},'
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    d_rl_loss,
                                                    d_fk_loss,
                                                    d_rl_only_loss,
                                                    d_fk_only_loss,
                                                    #d_nfk_loss,
                                                    g_adv_only_loss,
                                                    g_l1_only_loss,
                                                    g_adv_loss,
                                                    g_l1_loss,
                                                    hinge_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter / num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)

                    fdict = {self.gtruth_noisy[0]:sample_noisy,
                             self.zs1[0]:sample_z1,
                             self.zs2[0]:sample_z2}
                    
                    canvas_w, canvas_n = self.sess.run([self.Gs1[0], self.Gs2[0]],
                                                 feed_dict=fdict)
                    swaves = sample_wav
                    sample_dif = sample_wav - sample_noisy
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m,
                                                           np.max(canvas_w[m]),
                                                           np.min(canvas_w[m])))
                        wavfile.write(os.path.join(save_path,
                                                   'sample_{}-'
                                                   '{}.wav'.format(counter, m)),
                                      int(16e3),
                                      de_emph(canvas_w[m],
                                              self.preemph))
                        m_gtruth_path = os.path.join(save_path, 'gtruth_{}.'
                                                                'wav'.format(m))
                        if not os.path.exists(m_gtruth_path):
                            wavfile.write(os.path.join(save_path,
                                                       'gtruth_{}.'
                                                       'wav'.format(m)),
                                          int(16e3),
                                          de_emph(swaves[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'noisy_{}.'
                                                       'wav'.format(m)),
                                          int(16e3),
                                          de_emph(sample_noisy[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'dif_{}.wav'.format(m)),
                                          int(16e3),
                                          de_emph(sample_dif[m],
                                                  self.preemph))
                        np.savetxt(os.path.join(save_path, 'd_rl_losses.txt'),
                                   d_rl_losses)
                        np.savetxt(os.path.join(save_path, 'd_fk_losses.txt'),
                                   d_fk_losses)
                        np.savetxt(os.path.join(save_path, 'g_adv_losses.txt'),
                                   g_adv_losses)
                        np.savetxt(os.path.join(save_path, 'g_l1_losses.txt'),
                                   g_l1_losses)
                        np.savetxt(os.path.join(save_path, 'd_rl_only_losses.txt'),
                                   d_rl_only_losses)
                        np.savetxt(os.path.join(save_path, 'd_fk_only_losses.txt'),
                                   d_fk_only_losses)
                        np.savetxt(os.path.join(save_path, 'g_adv_only_losses.txt'),
                                   g_adv_only_losses)
                        np.savetxt(os.path.join(save_path, 'g_l1_only_losses.txt'),
                                   g_l1_only_losses)

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    # re-set batch idx
                    batch_idx = 0
                    # check if we have to deactivate L1
                    if curr_epoch >= config.l1_remove_epoch and self.deactivated_l1 == False:
                        print('** Deactivating L1 factor! **')
                        self.sess.run(tf.assign(self.l1_lambda, 0.))
                        self.deactivated_l1 = True
                    # check if we have to start decaying noise (if any)
                    if curr_epoch >= config.denoise_epoch and self.deactivated_noise == False:
                        # apply noise std decay rate
                        decay = config.noise_decay
                        if not hasattr(self, 'curr_noise_std'):
                            self.curr_noise_std = self.init_noise_std
                        new_noise_std = decay * self.curr_noise_std
                        if new_noise_std < config.denoise_lbound:
                            print('New noise std {} < lbound {}, setting 0.'.format(new_noise_std, config.denoise_lbound))
                            print('** De-activating noise layer **')
                            # it it's lower than a lower bound, cancel out completely
                            new_noise_std = 0.
                            self.deactivated_noise = True
                        else:
                            print('Applying decay {} to noise std {}: {}'.format(decay, self.curr_noise_std, new_noise_std))
                        self.sess.run(tf.assign(self.disc_noise_std, new_noise_std))
                        self.curr_noise_std = new_noise_std
                if curr_epoch >= config.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.join(threads)

    def clean(self, x):
        """ clean a utterance x
            x: numpy array containing the normalized noisy waveform
        """
        c_res = None
        for beg_i in range(0, x.shape[0], self.canvas_size):
            if x.shape[0] - beg_i  < self.canvas_size:
                length = x.shape[0] - beg_i
                pad = (self.canvas_size) - length
            else:
                length = self.canvas_size
                pad = 0
            x_ = np.zeros((self.batch_size, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]
            print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
            fdict = {self.gtruth_noisy[0]:x_}
            canvas_w = self.sess.run(self.Gs1[0],
                                     feed_dict=fdict)[0]
            canvas_w = canvas_w.reshape((self.canvas_size))
            print('canvas w shape: ', canvas_w.shape)
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                # get rid of last padded samples
                canvas_w = canvas_w[:-pad]
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        # deemphasize
        c_res = de_emph(c_res, self.preemph)
        return c_res    
