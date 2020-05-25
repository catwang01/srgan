#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D
from config import config
import argparse

###====================== HYPER-PARAMETERS ===========================###


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--lr_init', type=float, default=1e-4, help='lr_init of Adam')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
parser.add_argument('--n_epoch_init', type=int, default=100, help='intializer G')

parser.add_argument('--n_epoch', type=int, default=2000, help='adversarial learning SRGAN')
parser.add_argument('--lr_decay', type=float, default=0.1, help='lr_decay of adversarial learning')
parser.add_argument('--decay_every', type=float, default=100, help='decay for every')
parser.add_argument('--train_hr_img_path', type=str, default='DIV2K/DIV2K_train_HR/', help="train_hr_img_path")
parser.add_argument('--train_lr_img_path', type=str, default='DIV2K/DIV2K_train_LR_bicubic/X4/', help="train_lr_img_path")

parser.add_argument('--shuffle_buffer_size', type=int, default=128, help="shuffle buffer size")

parser.add_argument('--val_hr_img_path', type=str, default='DIV2K/DIV2K_valid_HR/', help="val_hr_img_path")
parser.add_argument('--val_lr_img_path', type=str, default='DIV2K/DIV2K_valid_LR_bicubic/X4/', help="val_lr_img_path")

## train set location

args = parser.parse_args()

tl.global_flag['mode'] = args.mode

## Adam

batch_size = args.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = args.lr_init
beta1 = args.beta1

## initialize G
n_epoch_init = args.n_epoch_init

## adversarial learning (SRGAN)
n_epoch = args.n_epoch
lr_decay = args.lr_decay
decay_every = args.decay_every
shuffle_buffer_size = args.shuffle_buffer_size

# create folders to save result images and trained models
save_dir = "samples"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)

def get_train_data():
    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))#[0:20]
        # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
        # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
        # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the entire train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
        # for im in train_hr_imgs:
        #     print(im.shape)
        # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
        # for im in valid_lr_imgs:
        #     print(im.shape)
        # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
        # for im in valid_hr_imgs:
        #     print(im.shape)
        
    # dataset API and augmentation
    def generator_train():
        for img in train_hr_imgs:
            yield img

    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [384, 384, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[96, 96])
        return lr_patch, hr_patch

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
        # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds

def train():
    G = get_G((batch_size, 96, 96, 3))
    D = get_D((batch_size, 384, 384, 3))
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    # change the mode to train
    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data()

    ## initialize learning (G)
    print("=" * 50 + " Initialize learning (G) " + '=' * 50)
    n_step_epoch = round(len(os.listdir(config.TRAIN.hr_img_path)) // batch_size)

    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{epoch}/{n_epoch_init}] step: [{step}/{n_step_epoch}] time: {time_elasped:.3f}s, mse: {mse_loss:.3f} ".format(
                epoch=epoch, n_epoch_init=n_epoch_init, step=step, n_step_epoch=n_step_epoch, time_elasped=time.time() - step_time, mse_loss=mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))

    ## adversarial learning (G, D)
    n_step_epoch = round(len(os.listdir(config.TRAIN.hr_img_path)) // batch_size)

    print("=" * 50 + " Training " + '=' * 50)
    for epoch in range(n_epoch):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if step == 2: break
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs+1)/2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print("Epoch: [{epoch}/{n_epoch}] step: [{step}/{n_step_epoch}] time: {time_elapsed:.3f}s, g_loss(mse:{mse_loss:.3f}, vgg:{vgg_loss:.3f}, adv:{g_gan_loss:.3f}) d_loss: {d_loss:.3f}".format(
                epoch=epoch, n_epoch = n_epoch, step= step, n_step_epoch= n_step_epoch, time_elapsed=time.time() - step_time, mse_loss=mse_loss, vgg_loss=vgg_loss, g_gan_loss=g_gan_loss, d_loss=d_loss))

        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g-{}-{}.h5'.format(epoch, mse_loss)))
            D.save_weights(os.path.join(checkpoint_dir, 'd-{}-{}.h5'.format(epoch, mse_loss)))

def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## if your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    G = get_G([1, None, None, 3])
    G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
    G.eval()

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
    tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))
    tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))



def main():

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")


if __name__ == '__main__':
    main()
