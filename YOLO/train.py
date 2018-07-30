from __future__ import print_function
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import gc

import dataset
from utils import *
from cfg import parse_cfg
from darknet import Darknet
import argparse

FLAGS = None
unparsed = None
device = None

use_cuda = None
eps = 1e-5
keep_backup = 5
save_interval = 2  # epoches
dot_interval = 70  # batches

# Test parameters
evaluate = False
conf_thresh = 0.25 # smaller confidence than this means we get rid of the prediction right away
nms_thresh = 0.4 # parameter of non-maximum suppression. The lower this value, the more liberal YOLO will find boxes next
                 # to each other.
iou_thresh = 0.5 # two bounding boxes with a larger IoU will count as positively localized


# Training settings
def load_testlist(testlist):
    """Load the dataloader of the test dataset from cards_data/cardsval.txt, allowing us to batch-process data"""
    init_width = model.width
    init_height = model.height

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist,
                            shape=(init_width, init_height),
                            shuffle=False,
                            transform=transforms.Compose([transforms.ToTensor(), ]), train=False),
                            batch_size=batch_size,
                            shuffle=False,
                            **kwargs)
    return loader


def main():
    """main method, containing training logic such as hardware, optimizer, weight loader, dataloaders etc.
       we make a number of variables available globally, as we will use them within the train method later on
    """
    datacfg = FLAGS.data
    cfgfile = FLAGS.config
    weightfile = FLAGS.weights

    data_options = read_data_cfg(datacfg)
    net_options = parse_cfg(cfgfile)[0]

    global use_cuda
    use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)

    globals()["trainlist"] = data_options['train']
    globals()["testlist"] = data_options['valid']
    globals()["backupdir"] = data_options['backup']
    globals()["gpus"] = data_options['gpus']  # e.g. 0,1,2,3
    globals()["ngpus"] = len(gpus.split(','))
    globals()["num_workers"] = int(data_options['num_workers'])

    globals()["batch_size"] = int(net_options['batch'])
    globals()["max_batches"] = int(net_options['max_batches'])
    globals()["learning_rate"] = float(net_options['learning_rate'])
    globals()["momentum"] = float(net_options['momentum'])
    globals()["decay"] = float(net_options['decay'])
    globals()["steps"] = [float(step) for step in net_options['steps'].split(',')]
    globals()["scales"] = [float(scale) for scale in net_options['scales'].split(',')]

    # Train parameters
    global max_epochs
    global batch_size
    global num_workers
    global max_batches
    if 'max_epochs' in net_options:
        max_epochs = int(net_options['max_epochs'])
    else:
        nsamples = file_lines(trainlist)
        max_epochs = (max_batches * batch_size) // nsamples + 1

    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    global model
    model = Darknet(cfgfile, use_cuda=use_cuda)
    model.load_weights(weightfile)
    # model.print_network()

    nsamples = file_lines(trainlist)


    init_epoch = model.seen // nsamples

    global loss_layers
    loss_layers = model.loss_layers
    for l in loss_layers:
        l.seen = model.seen

    globals()["test_loader"] = load_testlist(testlist)
    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay * batch_size}]
    global optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate / batch_size, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size)

    if evaluate:
        logging('evaluating ...')
        test(0)
    else:
        try:
            """here, the magic happens.
               we call train() every epoch and test() / savemodel() every few epochs
            """
            print("Training for ({:d}) epochs.".format(max_epochs))
            fscore = 0
            if init_epoch > save_interval:
                mfscore = test(init_epoch - 1)
            else:
                mfscore = 0.5
            for epoch in range(init_epoch, max_epochs):
                nsamples = train(epoch)
                if epoch > save_interval:
                    fscore = test(epoch)
                if (epoch + 1) % save_interval == 0:
                    savemodel(epoch, nsamples)
                if FLAGS.localmax and fscore > mfscore:
                    mfscore = fscore
                    savemodel(epoch, nsamples, True)
                print('-' * 90)
        except KeyboardInterrupt:
            print('=' * 80)
            print('Exiting from training by interrupt')


def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr


def curmodel():
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    return cur_model


def train(epoch):
    """training loop
    we iterate over the trainloader that has been loaded from cards_data/cardstrain.txt, allowing batch-processing
    """
    global processed_batches
    t0 = time.time()
    cur_model = curmodel() # return model from the main() loop above, containing all sorts of information that we gave it
    init_width = cur_model.width
    init_height = cur_model.height
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            train=True,
                            seen=cur_model.seen,
                            batch_size=batch_size,
                            num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    processed_batches = cur_model.seen // batch_size
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %e' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train() # enable train mode

    for batch_idx, (data, target) in enumerate(train_loader):
        """loop through dataloader, returning batches of size batch_size, usually set to 32"""
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        data, target = data.to(device), target.to(device) # send data and bounding boxes to GPU with CUDA if set to ON

        optimizer.zero_grad() # reset gradients

        output = model(data) # forward pass

        org_loss = []
        for i, l in enumerate(loss_layers):
            """accumulate loss over the three different loss layers (detection at three scales)"""
            l.seen = l.seen + data.data.size(0)
            ol = l(output[i]['x'], target)
            org_loss.append(ol)

        sum(org_loss).backward() # calculate gradients with backpropagation

        nn.utils.clip_grad_norm_(model.parameters(), 1000) # gradient clipping to prevent overflows etc.

        optimizer.step()# perform optimizer step

        del data, target
        org_loss.clear()
        gc.collect()

    print('')
    t1 = time.time()
    nsamples = len(train_loader.dataset)
    logging('training with %f samples/s' % (nsamples / (t1 - t0)))
    return nsamples


def savemodel(epoch, nsamples, curmax=False):
    """saving the weights of the network"""
    cur_model = curmodel()
    if curmax:
        logging('save local maximum weights to %s/localmax.weights' % (backupdir))
    else:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch + 1))
    cur_model.seen = (epoch + 1) * nsamples
    if curmax:
        cur_model.save_weights('%s/localmax.weights' % (backupdir))
    else:
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch + 1))
        old_wgts = '%s/%06d.weights' % (backupdir, epoch + 1 - keep_backup * save_interval)
        try:  # it avoids the unnecessary call to os.path.exists()
            os.remove(old_wgts)
        except OSError:
            pass


def test(epoch):
    """Test logic - one forward pass through the network is performed for a given batch
       using the evaluation methods from utils.py, we calculate precision, recall, and F-score
    """
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50

    model.eval()
    cur_model = curmodel()
    num_classes = cur_model.num_classes
    total = 0.0
    proposals = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            """loop through test_loader, returning batches of images and corresponding bounding boxes"""
            data = data.to(device)
            output = model(data)
            all_boxes = get_all_boxes(output, conf_thresh, num_classes, use_cuda=use_cuda)

            for k in range(data.size(0)):
                """loop through each image separately"""
                boxes = all_boxes[k]
                boxes = np.array(nms(boxes, nms_thresh))
                truths = target[k].view(-1, 5)
                num_gts = truths_length(truths)
                total = total + num_gts
                num_pred = len(boxes)

                if num_pred == 0:
                    continue

                proposals += int((boxes[:, 4] > conf_thresh).sum())
                for i in range(num_gts):
                    """loop over all ground truth bounding boxes"""
                    gt_boxes = torch.FloatTensor(
                        [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                    gt_boxes = gt_boxes.repeat(num_pred, 1).t()
                    pred_boxes = torch.FloatTensor(boxes).t()
                    # calculate all IoUs of a given GT bounding box and return the best ones
                    best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False), 0)
                    # pred_boxes and gt_boxes are transposed for torch.max
                    if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                        correct += 1

    precision = 1.0 * correct / (proposals + eps)
    recall = 1.0 * correct / (total + eps)
    fscore = 2.0 * precision * recall / (precision + recall + eps)
    logging("correct: %d, precision: %f, recall: %f, fscore: %f" % (correct, precision, recall, fscore))
    return fscore


if __name__ == '__main__':
    """exemplary usage: python train.py -d cards_data/cards.data -c cards_data/yolov3-tiny.cfg -w backup/000040.weights"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
                        type=str, default='cfg/sketch.data', help='data definition file')
    parser.add_argument('--config', '-c',
                        type=str, default='cfg/sketch.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w',
                        type=str, default='weights/yolov3.weights', help='initial weights file')

    FLAGS, _ = parser.parse_known_args()
    main()
