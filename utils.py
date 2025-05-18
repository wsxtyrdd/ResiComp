import logging
import os
from PIL import Image
import numpy as np
import torch


def save_config(config, file_path):
    import json
    info_json = json.dumps(config, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(file_path, 'w')
    f.write(info_json)


def logger_configuration(filename, phase, method='', save_log=True):
    logger = logging.getLogger(" ")
    workdir = './history/{}/{}'.format(method, filename)
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    if save_log:
        makedirs(workdir)
        makedirs(samples)
        makedirs(models)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, base_dir + "/checkpoint_best_loss.pth.tar")
    else:
        torch.save(state, base_dir + "/" + filename)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def load_weights(net, model_path, device, remove_keys=None):
    try:
        pretrained = torch.load(model_path, map_location=device)['state_dict']
    except:
        pretrained = torch.load(model_path, map_location=device)
    result_dict = {}
    for key, weight in pretrained.items():
        result_key = key
        load_flag = True
        if 'attn_mask' in key:
            load_flag = False
        if remove_keys is not None:
            for remove_key in remove_keys:
                if remove_key in key:
                    load_flag = False
        if load_flag:
            result_dict[result_key] = weight
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def read_image_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0) / 255
    return input_image
