#!/bin/bash

# Place the CUDA_VISIBLE_DEVICES="xxxx" required before the python call
# e.g. to specify the first two GPUs in your system: CUDA_VISIBLE_DEVICES="0,1" python ...

# SEGAN with no pre-emph and no bias in conv layers (just filters to downconv + deconv)
#CUDA_VISIBLE_DEVICES="2,3" python main.py --init_noise_std 0. --save_path segan_vanilla \
#                                          --init_l1_weight 100. --batch_size 100 --g_nl prelu \
#                                          --save_freq 50 --epoch 50

# SEGAN with pre-emphasis to try to discriminate more high freq (better disc of high freqs)
#CUDA_VISIBLE_DEVICES="1,2,3" python main.py --init_noise_std 0. --save_path segan_preemph \
#                                          --init_l1_weight 100. --batch_size 100 --g_nl prelu \
#                                          --save_freq 50 --preemph 0.95 --epoch 86

# Apply pre-emphasis AND apply biases to all conv layers (best SEGAN atm)
CUDA_VISIBLE_DEVICES="0,1" python main.py --init_noise_std 0. --save_path egan_alllbiased_preemph \
                                          --init_l1_weight 100. --batch_size 10 --g_nl prelu \
                                          --save_path /scratch3/sniu/segan/type_v \
                                          --synthesis_path /scratch3/sniu/segan/dwavegan_samples \
                                          --e2e_dataset /scratch3/sniu/data/segan.tfrecords \
                                          --save_clean_path /scratch3/sniu/segan/test_clean_results_v \
                                          --save_freq 200 --preemph 0.95 --bias_deconv True \
                                          --bias_downconv True --bias_D_conv True --decoder_type V \
                                          --weights /scratch3/sniu/segan/type_v/SEGAN_IV-9540 \
                                          --test_dir /scratch3/sniu/168_test_0db