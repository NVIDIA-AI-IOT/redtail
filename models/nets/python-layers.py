# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import caffe
from caffe.proto import caffe_pb2
import numpy as np
import random
import sys
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
from multiprocessing.dummy import Pool as ThreadPool

class BlankSquareLayer(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 1,         'requires a single layer.bottom'
        assert bottom[0].data.ndim >= 3, 'requires image data'
        assert len(top) == 1,            'requires a single layer.top'

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        # Copy all of the data
        top[0].data[...] = bottom[0].data[...]
        # Then zero-out one fourth of the image
        height = top[0].data.shape[-2]
        width = top[0].data.shape[-1]
        h_offset = random.randrange(height/2)
        w_offset = random.randrange(width/2)
        top[0].data[...,
                h_offset:(h_offset + height/2),
                w_offset:(w_offset + width/2),
                ] = 0

    def backward(self, top, propagate_down, bottom):
        pass

class TrailAugLayer(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 2,          'requires two bottom layers: data and label'
        assert bottom[0].data.ndim >= 3,  'requires image data'
        assert len(top) == 2,             'requires two top layers: data and label'
        assert len(bottom[1].shape) == 1, 'label dim be 1'
        self.index = 0
        # params is a python dictionary with layer parameters - see .prototxt files.
        params = eval(self.param_str)
        self.debugPrint = params['debug'] if 'debug' in params else False
        if self.debugPrint:
            print params
        self.is_hflip         = params['hflip']            if 'hflip' in params else False
        self.is_hflip3        = params['hflip3']           if 'hflip3' in params else False
        self.is_hflip5        = params['hflip5']           if 'hflip5' in params else False
        #self.is_hflip9        = params['hflip']            if 'hflip' in params else False
        self.contrastRadius   = params['contrastRadius']   if 'contrastRadius' in params else 0.0
        self.brightnessRadius = params['brightnessRadius'] if 'brightnessRadius' in params else 0.0
        self.saturationRadius = params['saturationRadius'] if 'saturationRadius' in params else 0.0
        self.sharpnessRadius  = params['sharpnessRadius']  if 'sharpnessRadius' in params else 0.0
        self.scaleMin         = params['scaleMin']         if 'scaleMin' in params else 1.0
        self.scaleMax         = params['scaleMax']         if 'scaleMax' in params else 1.0
        self.rotateAngle      = params['rotateAngle']      if 'rotateAngle' in params else 0.0
        self.topCut           = params['topCut']           if 'topCut' in params else 0.0
        self.blurProb         = params['blurProb']         if 'blurProb' in params else 0.0
        self.numThreads       = params['numThreads']       if 'numThreads' in params else 8

        # Thread pool for multithreading batch processing.
        self.pool = ThreadPool(self.numThreads)

        # List of enhancers.
        self.enhancers = [
            [ImageEnhance.Color,      self.saturationRadius],
            [ImageEnhance.Contrast,   self.contrastRadius],
            [ImageEnhance.Brightness, self.brightnessRadius],
            [ImageEnhance.Sharpness,  self.sharpnessRadius],
        ]

        # List of filters.
        self.filters = [
            [ImageFilter.GaussianBlur(1), self.blurProb],
        ]

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        # Shuffle enhancers and filters as these operations are not commutative.
        np.random.shuffle(self.enhancers)
        np.random.shuffle(self.filters)
        # Process images in parallel.
        batch_size = top[0].data.shape[0]
        res = self.pool.map(self.distort_image, zip(bottom[0].data, bottom[1].data, range(batch_size)))
        # Write the results.
        for i in range(top[0].data.shape[0]):
            top[0].data[i, ...] = res[i][0]
            top[1].data[i]      = res[i][1]
        self.index += batch_size

    def backward(self, top, propagate_down, bottom):
        pass

    def distort_image(self, item):
        def caffe_to_pil(img):
            # Original image is in CHW, BGR format. PIL works with HWC, RGB format.
            orig_shape = img.shape
            # Swap B and R channels and convert to HWC.
            img = img.reshape((orig_shape[0], -1))
            img = np.transpose(img[[2, 1, 0]]).reshape((orig_shape[1], orig_shape[2], orig_shape[0]))
            img = Image.fromarray(np.uint8(img))
            return img

        def pil_to_caffe(img):
            orig_shape = np.shape(img)
            # Transform image back to Caffe CHW, BGR format.
            img = np.reshape(img, (orig_shape[0] * orig_shape[1], orig_shape[2]))
            return np.float32(np.transpose(img)[[2, 1, 0]]).reshape((orig_shape[2], orig_shape[0], orig_shape[1]))

        def get_factor(radius):
            return np.random.uniform(1. - radius, 1. + radius)

        def cut_top(img, top_ratio):
            assert 0 <= top_ratio and top_ratio < 1
            # Just simple fair coin toss for now.
            # if top_ratio == 0 or np.random.binomial(1, 0.5) != 1:
            #     return img
            orig_size = img.size
            crop_size = (0, int(orig_size[1] * top_ratio), orig_size[0], orig_size[1])
            img = img.crop(crop_size)
            return img.resize(orig_size, resample=Image.BICUBIC)

        def scale_and_crop(img, min_scale, max_scale):
            assert min_scale <= max_scale
            orig_size = img.size
            if min_scale < max_scale:
                factor    = np.random.uniform(min_scale, max_scale)
                new_size  = (int(orig_size[0] * factor), int(orig_size[1] * factor))
                if factor > 1.:
                    # Resize the image first.
                    img = img.resize(new_size, resample=Image.BICUBIC)
                elif factor < 1.:
                    # Resize the image first.
                    img_r = img.resize(new_size, resample=Image.BICUBIC)
                    # Make the original image bigger to enable random crop.
                    img = img.resize((int(orig_size[0] * 1.1), int(orig_size[1] * 1.1)), resample=Image.BICUBIC)
                    x_offs = int((img.size[0] - img_r.size[0]) / 2)
                    y_offs = int((img.size[1] - img_r.size[1]) / 2)
                    # Paste resized image to the original one
                    # This provides a better way of padding, same as in rotate().
                    img.paste(img_r, (x_offs, y_offs))
            return crop(img, orig_size)

        def crop(img, image_crop_size):
            max_offsets = [img.size[0] - image_crop_size[0], img.size[1] - image_crop_size[1]]
            assert max_offsets[0] >= 0, 'Max crop x offset must be non-negative.'
            assert max_offsets[1] >= 0, 'Max crop y offset must be non-negative.'
            crop_x = np.random.randint(0, max_offsets[0]) if max_offsets[0] > 0 else 0
            crop_y = np.random.randint(0, max_offsets[1]) if max_offsets[1] > 0 else 0
            img = img.crop((crop_x, crop_y, crop_x + image_crop_size[0], crop_y + image_crop_size[1]))
            return img

        def rotate(img, delta_angle):
            if delta_angle == 0:
                return img
            # The main idea here is to rotate an image and use its expanded
            # version to do the padding (rather than using np.pad or similar methods).
            angle = np.random.randint(-delta_angle, delta_angle)
            # Rotate and expand to get non-clipped image size.
            img_r = img.rotate(angle, expand=True)
            # Resize to fill gaps.
            img_r = img.resize(img_r.size)
            # Paste original image to the center.
            x_offs = int((img_r.size[0] - img.size[0]) / 2)
            y_offs = int((img_r.size[1] - img.size[1]) / 2)
            img_r.paste(img, (x_offs, y_offs))
            # Rotate
            img_r = img_r.rotate(angle, resample=Image.BICUBIC, expand=False)
            # Crop rotated image.
            return img_r.crop((x_offs, y_offs, x_offs + img.size[0], y_offs + img.size[1]))

        def hflip(img, lab):
            # Flip image and label if necessary.
            # 3 classes
            if self.is_hflip3:
                if np.random.binomial(1, 0.5) != 1:
                    return img, lab
                # Flip left and right labels.
                if lab == 0:
                    lab = 2
                elif lab == 2:
                    lab = 0
                # Do horizontal flip of the image.
                img = ImageOps.mirror(img)
            elif self.is_hflip5:
                if np.random.binomial(1, 0.5) != 1:
                    return img, lab
                # Flip L_c and R_c or C_l and C_r
                label_remap = {0:4, 1:3, 2:2, 3:1, 4:0}
                lab = label_remap[lab]
                # Do horizontal flip of the image.
                img = ImageOps.mirror(img)
            elif self.is_hflip:
                # Just flip the image.
                if np.random.binomial(1, 0.5) == 1:
                    img = ImageOps.mirror(img)
            return img, lab

        img, lab, idx = item
        orig_shape = img.shape
        img = caffe_to_pil(img)
        if self.debugPrint:
            print "Original label: " + str(lab)
            img.save('/data/tmp/' + str(self.index) + '_orig.jpg')
        # Cut top part of the image.
        img = cut_top(img, self.topCut)
        # Scale and crop.
        img = scale_and_crop(img, self.scaleMax, self.scaleMax)
        # Rotate.
        img = rotate(img, self.rotateAngle)
        # Horizontal flip.
        img, lab = hflip(img, lab)
        # Apply filters.
        for cur_f in self.filters:
            f = cur_f[0]
            p = cur_f[1]
            if np.random.uniform() < p:
                img = img.filter(f)
        # Apply color, brightness etc transforms.
        for cur_e in self.enhancers:
            e   = cur_e[0](img)
            img = e.enhance(get_factor(cur_e[1]))
        # Write results.
        res_img = pil_to_caffe(img)
        res_lab = lab
        if self.debugPrint:
            caffe_to_pil(res_img).save('/data/tmp/' + str(self.index) + '_result.jpg')
            print "Result label  : " + str(res_lab)
        return (res_img, res_lab)

# Loss layer for cross entropy with softmax and entropy optimization.
class CrossEntropySoftmaxWithEntropyLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires 2 bottom layers'
        assert len(top)    == 1, 'requires a single top layer'
        #sys.stdout.flush()

        # params is a python dictionary with layer parameters - see .prototxt files.
        params = eval(self.param_str)
        self.ent_scale = params['entScale'] if 'entScale' in params else 0.01
        self.p_scale   = params['pScale']   if 'pScale'   in params else 0.0001
        self.label_eps = params['label_eps'] if 'label_eps' in params else 0.0

    def reshape(self, bottom, top):
        # Reshape top blob as scalar.
        top[0].reshape(1)

    def forward(self, bottom, top):
        lgt_blob = bottom[0].data
        lab_blob = bottom[1].data
        res_blob = top[0].data
        total_loss = 0.0
        smooth_lab  = np.zeros(3)
        smooth_val  = self.label_eps / (len(smooth_lab) - 1)
        for i in range(bottom[0].data.shape[0]):
            lab = int(lab_blob[i])
            lgt = lgt_blob[i]
            sm   = self.softmax(lgt)
            lse  = self.log_sum_exp(lgt)
            smooth_lab.fill(smooth_val)
            smooth_lab[lab] = 1.0 - self.label_eps
            ce   = -np.sum(smooth_lab * (lgt - lse))
            ent  = -np.sum(sm * (lgt - lse))
            loss = ce - self.ent_scale * ent
            scale = [self.p_scale, 0.0, self.p_scale]
            loss += scale[lab] * sm[2 - lab]
            total_loss += loss
        res_blob[0] = total_loss / bottom[0].data.shape[0]

    def backward(self, top, propagate_down, bottom):
        lgt_blob = bottom[0].data
        lab_blob = bottom[1].data
        lgt_diff = bottom[0].diff
        smooth_lab  = np.zeros(3)
        smooth_val  = self.label_eps / (len(smooth_lab) - 1)
        for i in range(bottom[0].data.shape[0]):
            lab = int(lab_blob[i])
            lgt = lgt_blob[i]
            sm  = self.softmax(lgt)
            log_sm = self.log_softmax(lgt)
            smooth_lab.fill(smooth_val)
            smooth_lab[lab] = 1.0 - self.label_eps
            a = np.sum((1.0 + log_sm) * sm) - 1.0
            ent_diff    = sm * (a - log_sm)
            lgt_diff[i] = (sm - smooth_lab) - self.ent_scale * ent_diff
            scale                = [self.p_scale, 0.0, self.p_scale]
            lgt_diff             -= scale[lab] * sm[2 - lab] * sm
            lgt_diff[i, 2 - lab] += scale[2 - lab] * sm[2 - lab]
        lgt_diff /= bottom[0].data.shape[0]

    def softmax(self, logits):
        e = np.exp(logits - np.max(logits))
        return e / np.sum(e)

    def log_sum_exp(self, x):
        a = np.max(x)
        return a + np.log(np.sum(np.exp(x - a)))

    def log_softmax(self, logits):
        return logits - self.log_sum_exp(logits)
