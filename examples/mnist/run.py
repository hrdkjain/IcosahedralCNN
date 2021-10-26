import argparse
import gzip
import os
import pickle
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data_utils
from tqdm import tqdm
import itertools

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# https://stackoverflow.com/questions/60730544/tensorboard-colab-tensorflow-api-v1-io-gfile-has-no-attribute-get-filesystem
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../../")))
import icocnn
import misc
from model import IcoConvNet_OriginalR2R

MNIST_R_R_PATH = "data/ico_4_mnist_r_r.gz"
MNIST_NR_NR_PATH = "data/ico_4_mnist_nr_nr.gz"
MNIST_NR_R_PATH = "data/ico_4_mnist_nr_r.gz"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params = {}
params['data_dir'] = MNIST_R_R_PATH
params['log_dir'] = 'log/test'
params['epochs'] = 60
params['batch_size'] = 256
params['print_freq'] = 2
params['resume'] = ''
params['start_epoch'] = 0
params['gpu'] = 0
params['learning_rate'] = 5e-3
params['debug'] = False
params['use_datasubset'] = False    #To use data subset of quick checks
params['log_mesh_epoch'] = 2    #Specify the frequency of logging mesh, projector and images
params['corner_mode'] = 'zeros'

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
classes_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def load_data():
    with gzip.open(params['data_dir'], 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(dataset["train"]["labels"].astype(np.int64))

    # normalization
    train_data = train_data / 255.0
    mean = train_data.mean()
    stdv = train_data.std()
    train_data = (train_data - mean) / stdv

    if params['use_datasubset']:
        train_data = train_data[0:324]
        train_labels = train_labels[0:324]
    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    val_data = torch.from_numpy(dataset["test"]["images"][:, None, :, :].astype(np.float32))
    val_labels = torch.from_numpy(dataset["test"]["labels"].astype(np.int64))

    # normalization
    val_data = val_data / 255.0
    val_data = (val_data - mean) / stdv

    if params['use_datasubset']:
        val_data = val_data[0:324]
        val_labels = val_labels[0:324]
    val_dataset = data_utils.TensorDataset(val_data, val_labels)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True)

    params['trn_size'] = len(train_dataset)
    params['val_size'] = len(val_dataset)
    params['iters_per_train_epoch'] = len(train_loader)
    params['iters_per_val_epoch'] = len(val_loader)
    return train_loader, val_loader, train_dataset, val_dataset

def main():
    global params
    train_loader, val_loader, train_dataset, _ = load_data()

    writer = SummaryWriter(log_dir=params['log_dir'])

    model = IcoConvNet_OriginalR2R(params['corner_mode'])
    model.to(DEVICE)
    images, labels = next(iter(train_loader))
    images = images.cuda(params['gpu'], non_blocking=True)
    writer.add_graph(model, images)
    writer.flush()

    print("#parameters", sum(x.numel() for x in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    best_acc1 = 0
    # optionally resume from a checkpoint
    if params['resume']:
        if os.path.isfile(params['resume']):
            print("=> loading checkpoint '{}'".format(params['resume']))
            if params['gpu'] is None:
                checkpoint = torch.load(params['resume'])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(params['gpu'])
                checkpoint = torch.load(params['resume'], map_location=loc)
            params['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if params['gpu'] is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(params['gpu'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(params['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(params['resume']))

    cudnn.benchmark = True
    misc.save_params(params, sys.argv[1:])

    # plot_filters(model, writer)

    # Create log data which is used for validation, shouldnt be modified over the epochs
    log_data = {}
    if params['log_mesh_epoch']:
        images, labels = next(iter(val_loader))
        log_data['images'], log_data['labels'] = images, labels
        if params['gpu'] is not None:
            log_data['images'] = log_data['images'].cuda(params['gpu'], non_blocking=True)
        log_data['labels'] = log_data['labels'].cuda(params['gpu'], non_blocking=True)
        log_data['images_subset'], log_data['labels_subset'] = select_n_random(log_data['images'], log_data['labels'], n=3)

    for epoch in tqdm(range(params['start_epoch'], params['epochs'])):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, writer, log_data)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save checkpoint
        misc.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, params['log_dir'])

    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = misc.AverageMeter('Time', ':6.3f')
    data_time = misc.AverageMeter('Data', ':6.3f')
    losses = misc.AverageMeter('Loss', ':.4e')
    top1 = misc.AverageMeter('Acc@1', ':6.2f')
    top5 = misc.AverageMeter('Acc@5', ':6.2f')
    progress = misc.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if params['gpu'] is not None:
            images = images.cuda(params['gpu'], non_blocking=True)
            labels = labels.cuda(params['gpu'], non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        acc1, acc5 = misc.accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))            # measure accuracy and record loss

        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % params['print_freq'] == 0:
            if params['debug']:
                progress.display(i)
            writer.add_scalars('Accuracy', {'trn_top1': acc1[0], 'trn_top5': acc5[0]},
                               epoch * params['trn_size'] + i * params['batch_size'])
            writer.add_scalars('Loss', {'trn': loss.item()}, epoch * params['trn_size'] + i * params['batch_size'])
            writer.flush()

def validate(val_loader, model, criterion, epoch, args, writer, log_data):
    batch_time = misc.AverageMeter('Time', ':6.3f')
    losses = misc.AverageMeter('Loss', ':.4e')
    top1 = misc.AverageMeter('Acc@1', ':6.2f')
    top5 = misc.AverageMeter('Acc@5', ':6.2f')
    progress = misc.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        confusion_matrix = torch.zeros([len(classes), len(classes)], dtype=torch.long)
        for i, (images, labels) in enumerate(val_loader):
            if params['gpu'] is not None:
                images = images.cuda(params['gpu'], non_blocking=True)
                labels = labels.cuda(params['gpu'], non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = misc.accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute confusion matrix
            _, preds = torch.max(output, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if params['debug'] and i % params['print_freq'] == 0:
                progress.display(i)

        writer.add_scalars('Accuracy', {'val_top1': top1.avg, 'val_top5': top5.avg}, (epoch + 1) * params['trn_size'])
        writer.add_scalars('Loss', {'val': losses.avg}, (epoch + 1) * params['trn_size'])



        if params['log_mesh_epoch'] and epoch % params['log_mesh_epoch'] == 0:
            # log intermediate output, images and embedding on validation data

            # plot confusion matrix
            fig = plt.figure(figsize=(10, 10))
            plot_confusion_matrix(confusion_matrix, classes)
            writer.add_figure('ConfusionMatrix', fig, (epoch + 1))

            # plot filters
            # plot_filters(model, writer, (epoch + 1))

            # log meshes
            # hooks = {}
            # for name, m in model.named_modules():
            #     if isinstance(m, icocnn.ico_conv._IcoConv):
            #         tag = m._get_name() + '_fi' + str(m.in_features) + '_fo' + str(m.out_features) + \
            #               '_su' + str(m.in_subdivisions) + '_st' + str(m.stride)
            #         hooks[name] = m.register_forward_hook(log_mesh(writer, tag, range(6), 0))
            # model(log_data['images_subset'])
            # for h in hooks.values():
            #     h.remove()

            # Images
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(log_data['images_subset'], model(log_data['images_subset']),
                                                 log_data['labels_subset']), global_step=epoch)

            # Embedding Projector on the log data
            writer.add_embedding(model(log_data['images']), metadata=log_data['labels'], global_step=epoch)

        writer.flush()

        # TODO: this should also be done with the misc.ProgressMeter
        tqdm.write(
            'Epoch:{epoch}, top1Acc:{top1.avg:.2f}%, top5Acc:{top5.avg:.2f}%'.format(epoch=epoch, top1=top1, top5=top5))

    return top1.avg

def matplotlib_imshow(img, one_channel=False):
    # helper function to show an image
    # (used in the `plot_classes_preds` function below)
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def select_n_random(data, labels, output=[], n=3):
    # helper function
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)
    if not output==[]:
        assert len(data) == len(output)

    perm = torch.randperm(len(data))
    if output==[]:
        return data[perm][:n], labels[perm][:n]
    else:
        return data[perm][:n], labels[perm][:n], output[perm][:n]

def images_to_probs(output):
    '''
    Generates predictions and corresponding probabilities from out of a trained network
    '''
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(images, output, labels, n=3):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(output)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure()  # figsize=(12, 48)
    for idx in np.arange(n):
        ax = fig.add_subplot(1, n, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.ylim(-0.5, len(classes) - 0.5)

def show(img):
    npimg = img.cpu().detach().numpy()
    temp = np.transpose(npimg, (1, 2, 0))
    plt.imshow(temp[:, :, 0], cmap='coolwarm', interpolation='nearest', vmin=-1.5, vmax=1.5)
    plt.colorbar()

def plot_filters(model, summarywriter=None, epoch=-1):
    import icocnn
    import torchvision

    plot_backend = plt.get_backend()
    plt.switch_backend('agg')

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, icocnn.ico_conv._IcoConv):
                filters = m.weight.new_zeros([m.out_channels, m.in_channels, 3, 3])
                bias_expanded = None
                if m.bias is not None:
                    bias_expanded = m.weight.new_empty(m.out_channels)
                m.compose_filters(filters, bias_expanded)

                f = filters[:12,:12].reshape(-1, 1, 3, 3)

                img = torchvision.utils.make_grid(f, 12, normalize=False)
                fig = plt.figure()
                show(img)
                if summarywriter:
                    tag = m._get_name() + '_fi' + str(m.in_features) + '_fo' + str(m.out_features) + \
                          '_su' + str(m.in_subdivisions) + '_st' + str(m.stride)
                    summarywriter.add_figure(tag, fig, epoch)

    plt.switch_backend(plot_backend)

def log_mesh(writer, tag, channels, epoch):
    def hook(model, input, output):
        for c in channels:
            v, f, v_colors = icocnn.utils.ico_geometry.imgbatch_to_icomesh(output[:, c:c+1], highlight_boundaries=True)
            writer.add_mesh(tag + '_c' + str(c), vertices=v, colors=v_colors, faces=f, global_step=epoch)
    return hook

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=params['data_dir'], metavar='DIR',
                        help='path to dataset (default: {})'.format(params['data_dir']))
    parser.add_argument('--log_dir', default=params['log_dir'], metavar='DIR',
                        help='path to log the experiment (default: {})'.format(params['log_dir']))
    parser.add_argument('--epochs', default=params['epochs'], type=int, metavar='N',
                        help='number of total epochs to run (default: {0})'.format(params['epochs']))
    parser.add_argument('-b', '--batch_size', default=params['batch_size'], type=int, metavar='N',
                        help='mini-batch size (default: {0})'.format(params['batch_size']))
    parser.add_argument('-p', '--print_freq', default=params['print_freq'], type=int, metavar='N',
                        help='print frequency defined by number of batches (default: {0})'.format(params['print_freq']))
    parser.add_argument('--resume', default=params['resume'], type=str, metavar='PATH',
                        help='path to latest checkpoint (default: {})'.format(params['resume']))
    parser.add_argument('--start_epoch', default=params['start_epoch'], type=int, metavar='N',
                        help='manual epoch number, useful on restarts (default: {0})'.format(params['start_epoch']))
    parser.add_argument('--gpu', default=params['gpu'], type=int, metavar='N',
                        help='GPU id to use (default: {})'.format(params['gpu']))
    parser.add_argument('--debug', default=params['debug'], type=bool, const=True, nargs='?',
                        help='Activate nice mode (default: {})'.format(params['debug']))
    parser.add_argument('--corner_mode', default=params['corner_mode'], metavar='DIR',
                        help='Setting corners to (zeros, average). (default: {})'.format(params['corner_mode']))

    args = parser.parse_args()
    for arg in vars(args):
        params[arg] = getattr(args, arg)

    assert params['corner_mode'] == 'zeros' or params['corner_mode'] == 'average'
    main()
