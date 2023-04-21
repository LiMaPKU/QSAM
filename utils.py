import logging
from logging.handlers import TimedRotatingFileHandler
import io
import PIL
import PIL.Image as Image
import warnings
import torch
from torchvision.transforms import functional as F
import random
import math
import numbers
import collections
import numpy as np

zigzag_index = (
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
)


def generate_qim(qtable, im_size=32):
    tmp = np.hstack([qtable for i in range(im_size // 8)])
    return np.vstack([tmp for i in range(im_size // 8)])


def get_qim(jpg_quality):
    qims = []
    buffer = io.BytesIO()
    img = Image.open('./jpg.jpg')
    img.save(buffer, format='jpeg', quality=jpg_quality)
    coded_im = Image.open(buffer)
    qt = coded_im.quantization
    qims.append(convert_dict_qtables(qt))
    return qims


def convert_dict_qtables(qtables):
    assert len(qtables) == 2
    qtables = [qtables[key] for key in range(len(qtables)) if key in qtables]
    for idx, table in enumerate(qtables):
        qtables[idx] = np.array([table[i] for i in zigzag_index]).reshape(8, 8)
    return np.concatenate([np.expand_dims(qtable, axis=0) for qtable in qtables], axis=0)


class AddMultiJpgNoise(object):
    def __init__(self, jpg_qualities):
        self.jpg_qualities = jpg_qualities

    def __call__(self, img):
        coded_ims = []
        for quality in self.jpg_qualities:
            buffer = io.BytesIO()
            if quality == 0:
                coded_ims.append(img)
                continue
            elif quality < 101:
                img.save(buffer, format='jpeg', quality=quality)
            else:
                raise NotImplementedError
            coded_im = Image.open(buffer)
            coded_ims.append(coded_im)
        return coded_ims


class AddJpgNoise(object):
    def __init__(self, jpg_quality):
        self.jpg_quality = jpg_quality

    def __call__(self, img):
        if self.jpg_quality == 0:
            return img
        buffer = io.BytesIO()
        if self.jpg_quality < 101:
            img.save(buffer, format='jpeg', quality=self.jpg_quality)
        else:
            raise NotImplementedError
        coded_im = Image.open(buffer)
        return coded_im


class GetQim(object):
    def __call__(self, pic):
        if isinstance(pic, PIL.PngImagePlugin.PngImageFile):
            return [pic, np.ones([2, 64])]
        else:
            return [pic, convert_dict_qtables(pic.quantization)]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomAddJpgNoise(object):
    def __init__(self, jpg_qualities):
        self.jpg_qualities = jpg_qualities

    def __call__(self, img):
        self.jpg_quality = random.choice(self.jpg_qualities)
        if self.jpg_quality == 0:
            return img
        buffer = io.BytesIO()
        if self.jpg_quality < 101:
            img.save(buffer, format='jpeg', quality=self.jpg_quality)
        else:
            raise NotImplementedError
        coded_im = Image.open(buffer)
        return coded_im


def get_logger(name, log_dir='log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = log_dir + '{}.info.log'.format(name)
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    error_name = log_dir + '{}.error.log'.format(name)
    error_handler = TimedRotatingFileHandler(error_name,
                                             when='D',
                                             encoding='utf-8')
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_target = self.next_target.cuda(non_blocking=True)
            if isinstance(self.next_input, list):
                self.next_input = [next_in.cuda(non_blocking=True) for next_in in self.next_input]
                self.next_input = [next_in.float() for next_in in self.next_input]
            else:
                self.next_input = self.next_input.cuda(non_blocking=True)
                self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
Sequence = collections.abc.Sequence


class FormerToRGB(object):
    def __init__(self):
        pass

    def __call__(self, img_list):
        return [img_list[0].convert('RGB'), img_list[1]]


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class MultiRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img_list, scale, ratio):
        width, height = _get_image_size(img_list[0])
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img_list):
        i, j, h, w = self.get_params(img_list, self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in img_list]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class MultiRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_list):
        if random.random() < self.p:
            return [F.hflip(img) for img in img_list]
        return img_list

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class FormerRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_list):
        if random.random() < self.p:
            return [F.hflip(img_list[0]), img_list[1]]
        return img_list

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class MultiToTensor(object):
    def __call__(self, pic_list):
        return [F.to_tensor(pic) for pic in pic_list]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AllToTensor(object):
    def __call__(self, pic_list):
        return [F.to_tensor(pic_list[0]), F.to_tensor(pic_list[1]).float()]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MultiNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor_list):
        return [F.normalize(tensor, self.mean, self.std, self.inplace) for tensor in tensor_list]

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NormalizeWithoutQims(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor_list):
        return [F.normalize(tensor_list[0], self.mean, self.std), tensor_list[1]]

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MultiPad(object):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img_list):
        return [F.pad(img, self.padding, self.fill, self.padding_mode) for img in img_list]

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)


class FormerPad(object):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img_list):
        return [F.pad(img_list[0], self.padding, self.fill, self.padding_mode), img_list[1]]

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)


class MultiRandomCrop(object):
    def __init__(self, size, fill=0, padding_mode='constant', padding=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
        self.padding_mode = padding_mode
        self.padding = padding

    @staticmethod
    def get_params(img_list, output_size):
        w, h = _get_image_size(img_list[0])
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_list):
        i, j, h, w = self.get_params(img_list, self.size)
        return [F.crop(img, i, j, h, w) for img in img_list]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class FormerRandomCrop(object):
    def __init__(self, size, fill=0, padding_mode='constant', padding=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
        self.padding_mode = padding_mode
        self.padding = padding

    @staticmethod
    def get_params(img_list, output_size):
        w, h = _get_image_size(img_list[0])
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_list):
        i, j, h, w = self.get_params(img_list, self.size)
        return [F.crop(img_list[0], i, j, h, w), img_list[1]]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def find_index(seq, item):
    for i, x in enumerate(seq):
        if item == x:
            return i
    return -1


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep, logger):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 **
                              (float(ep + 1 - start_decay_at_ep) / (150 - start_decay_at_ep))))
    logger.info('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))
