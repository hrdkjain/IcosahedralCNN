import os
import shutil
import torch

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from icocnn.utils.ico_geometry import get_mesh_from_icomapping


def save_params(params, sys_args):
    if not os.path.exists(os.path.join(params['log_dir'])):
        os.makedirs(os.path.join(params['log_dir']))

    with open(os.path.join(params['log_dir'], 'params.txt'), 'w+') as f:
        for key in params.keys():
            value = params[key]
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                f.write('%s: %s\n' % (key, str(value)))
            elif isinstance(value, list):
                f.write('%s:\n' % key)
                for v in value:
                    f.write(v + '\n')

    with open(os.path.join(params['log_dir'], 'args.txt'), 'w') as f:
        f.write('\n'.join(sys_args))


def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(log_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(log_dir, filename), os.path.join(log_dir, 'model_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        element_cnt = target.nelement()

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.transpose(0, 1)
        correct = pred.eq(target.expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / element_cnt))
        return res


def iou_score(pred_cls, true_cls, nclass=3):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    with torch.no_grad():
        iou = []
        for i in range(nclass):
            # intersect = ((pred_cls == i) + (true_cls == i)).eq(2).item()
            # union = ((pred_cls == i) + (true_cls == i)).ge(1).item()
            intersect = ((pred_cls == i).type(torch.int32) + (true_cls == i).type(torch.int32)).eq(2).sum().item()
            union = ((pred_cls == i).type(torch.int32) + (true_cls == i).type(torch.int32)).ge(1).sum().item()
            iou_ = float(intersect) / float(union)
            iou.append(iou_)
        return np.array(iou)


def average_precision(score_cls, true_cls, classes):
    with torch.no_grad():
        score = score_cls.cpu().numpy()
        score = score.reshape(score.shape[0], score.shape[1], -1)
        true = label_binarize(true_cls.cpu().numpy().reshape(-1), classes=classes)
        score = np.swapaxes(score, 1, 2).reshape(-1, len(classes))
        return average_precision_score(true, score)


def per_class_accuracy(pred_cls, true_cls, nclass=3):
    """
    compute per-node classification accuracy
    """
    with torch.no_grad():
        accu = []
        for i in range(nclass):
            intersect = ((pred_cls == i).type(torch.int32) + (true_cls == i).type(torch.int32)).eq(2).sum().item()
            thiscls = (true_cls == i).type(torch.int32).sum().item()
            accu.append(float(intersect) / float(thiscls))
        return np.array(accu)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logfile=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logfile = logfile

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = '\t'.join(entries)
        self.display_message(message)

    def display_message(self, message):
        print(message, flush=True)
        if self.logfile:
            postfix = '' if message[-2:] == '\n' else '\n'
            self.logfile.write(message + postfix)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def log_mesh(writer, tag, outputs, targets, epoch, max_distance=-1.0, interpolate_poles=True):
    v, f, v_colors = get_mesh_from_icomapping(outputs, targets, max_distance, interpolate_poles)
    writer.add_mesh(tag, vertices=v, colors=v_colors, faces=f, global_step=epoch)
