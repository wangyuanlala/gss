from datetime import datetime
from torch.utils.data import DataLoader
from GSS.resnet import *
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from GSS.s_distance import *
from torch.autograd import Variable
import dataloader_cifar as dataloader
from torch.nn.parameter import Parameter
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
import copy


parser = argparse.ArgumentParser()

parser.add_argument('-MODEL', default='ResNet18')
parser.add_argument('--dataset', default='cifar10', help='dataset')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate for encoder', dest='lr')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--mixup', default=False, type=bool, help='use mixup')
parser.add_argument('--data_split', default='imbalance', type=str, help='data split type')

args = parser.parse_args()  


args.epochs = 100
args.cos = False
args.schedule = []  # cos in use
args.symmetric = False

ot_criterion = SinkhornDistance_uniform(eps=0.1, max_iter=200, reduction='mean').to('cuda')


def OT_consistency_2(feats, labels, prototypes, prior_weights=None):

    if prior_weights is None:
        nu_prior = None
    else:
        nu_prior = prior_weights.softmax(-1)
    
    _ot_loss3, _pi3 = ot_criterion(feats, prototypes, nu=nu_prior)
    p_labels3 = torch.argmax(_pi3.log().softmax(-1), -1)

    return _ot_loss3, p_labels3, p_labels3 == labels, _pi3.log().softmax(-1)


def warmup(epoch, net, optimizer, dataloader, criterion):
    net.train()
    
    loss, train_bar = 0.0, tqdm(dataloader)
    total = 0.0
    correct = 0.0

    for imgs, labels, index in train_bar:
        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        
        feats, outputs = net(imgs)
           
        loss = criterion(outputs, labels).mean()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        
        loss.backward()  
        optimizer.step() 

        train_bar.set_description('Warm-up Epoch: [{}/{}], Train Acc: {:.6f}, Loss-1: {:.4f}'.format(
                                epoch, 100, 100.*correct/total, loss.item()))
                                

def gss(net, data_loader, epoch, args):
    net.encoder.eval()
    net.classifier.eval()

    eval_bar = tqdm(data_loader)
    
    train_size = data_loader.dataset.train_label.shape[0]
    
    total_features = torch.zeros((train_size, 512))
    total_labels = torch.zeros(train_size).long()

    unique_labels = torch.linspace(0, 9, 10).long()
    
    group_features = {}

    for ul in unique_labels:
        group_features.update({str(ul.cpu().numpy()): []})
    
    with torch.no_grad():
        for imgs, labels, index in eval_bar:
            
            imgs, labels = imgs.cuda(), labels.cuda() 

            feats, _ = model(imgs)
            
            for b, (lab, fs) in enumerate(zip(labels, feats)):
                group_features[str(lab.cpu().numpy())].append(fs.cpu().numpy())

                total_features[index[b]] = feats[b]
                total_labels[index[b]] = labels[b]

            eval_bar.set_description('Eval Epoch: [{}/{}]'.format(epoch, args.epochs))
    
    avg_features = torch.zeros((unique_labels.size(0), 512))
    
    for keys, values in group_features.items():
        if not values.__len__() == 0:
            avg_features[int(keys)] = torch.from_numpy(np.array(values)).squeeze().mean(0)
    
    avg_features = avg_features.cuda()

    return avg_features


def train_e(net, data_loader, train_optimizer, criterion, scheduler, epoch, avg_prototypes, args):

    loss, train_bar = 0.0, tqdm(data_loader)
    total = 0.0
    correct = 0.0
    total_2 = 0.0

    total_p_true = 0
    total_p_equals_given = 0

    true_y = []
    true_x = []
    true_y_prob = []

    for imgs, labels, index in train_bar:
        train_optimizer.zero_grad()

        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
 
        net.encoder.eval()
        net.classifier.eval()
        feats, _ = net(imgs)

        consistency_loss, ot_pseudo_labels, consistency_index, p_prob = OT_consistency_2(feats, labels, avg_prototypes, prior_weights=None)

        true_index = torch.nonzero(consistency_index).squeeze()
        true_label = ot_pseudo_labels[true_index]
        true_x_index = index[true_index]
        true_label_prob = p_prob[true_index]

        if len(true_index.shape) == 0 or true_index.shape[0] <= 3:
            continue

        true_x.append(true_x_index.cpu().detach())
        true_y.append(true_label.cpu().detach())
        true_y_prob.append(true_label_prob.cpu().detach())

        net.encoder.train()
        net.classifier.train()
        
        inputs, _labels = pad_to_multiple_of_8(imgs[true_index], true_label)

        _, outputs = net(inputs)

        l1 = criterion(outputs, _labels).sum() / labels.size(0)
        l2 = l1

        loss = l1 
            
        loss.backward()
        train_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(_labels.data).cpu().sum()
        
        total_p_true += torch.nonzero(ot_pseudo_labels == _clean_labels[index]).size(0)

        total_2 += (true_label == _clean_labels[true_x_index]).size(0)
        total_p_equals_given += torch.nonzero(true_label == _clean_labels[true_x_index]).size(0)

        train_bar.set_description('Estimating Epoch: [{}/{}], Train Acc: {:.6f}, ot_true: {:.6f}, ot_eq_given: {:.6f}, Loss-1: {:.4f}, Loss-2: {:.4f}'.format(
                                epoch, args.epochs, 100.*correct/total, 100.*total_p_true/total, 100.*total_p_equals_given/total_2, l1.item(), l2.item()))
    
    scheduler.step()

    try:
        true_x = torch.cat(true_x).cpu().numpy()
        true_y = torch.concat(true_y).cpu().numpy()
        true_y_prob = torch.concat(true_y_prob).cpu().numpy()
    except:
        import pdb;pdb.set_trace()
    
    return loss.item(), [true_x, true_y, true_y_prob]

# train for one epoch
def train_fc(net, data_loader, train_optimizer, criterion, epoch, args):
    net.encoder.train()
    net.classifier.train()

    loss, train_bar = 0.0, tqdm(data_loader)
    total = 0.0
    correct = 0.0

    for imgs, labels, index in train_bar:
        train_optimizer.zero_grad()

        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        if args.mixup and not args.cutmix:
            inputs, targets_a, targets_b, lam = mixup_data(imgs, labels, 0.5, use_cuda=True)

            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            feats, outputs = net(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
        
        
        lam = np.random.beta(0.2, 0.2)
        rand_index = torch.randperm(imgs.size()[0]).cuda()

        targets_a = labels
        targets_b = labels[rand_index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
        # # compute output
        feats, outputs = net(imgs)
                
        loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1. - lam)
        # loss = criterion(outputs, targets_a) 

        loss = loss.mean()

        loss.backward()
        train_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()    
        
        train_bar.set_description('Train FC: [{}/{}], Train Acc: {:.6f}, Loss-1: {:.4f}'.format(epoch, args.epochs, 100.*correct/total, loss.item()))   
    

def test(net, data_loader, criterion, epoch, args, avg_prototypes=None,):
    net.eval()
    correct = 0
    correct_ot = 0

    total = 0
    total_preds = []
    total_targets = []
    total_test_loss = []
    
    loss, test_bar = 0.0, tqdm(data_loader)

    with torch.no_grad():
        for imgs, labels in test_bar:

            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            feats, outputs = net(imgs)

            _, predicted = torch.max(outputs, 1)  

            correct_ot += predicted.eq(labels).cpu().sum().item()                 
          
            correct += predicted.eq(labels).cpu().sum().item()                 

            total_preds.append(predicted)
            total_targets.append(labels)

            test_loss = criterion(outputs, labels)
            total_test_loss.append(test_loss)
            
            total += imgs.size(0)

            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% OT-Acc@1:{:.2f}%'.format(epoch, args.epochs, correct / total * 100, correct_ot / total * 100))

    acc = 100.*correct/total

    acc_ot = 100.*correct_ot/total

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    cls_acc = [ round( 100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(10)]
    
    return acc, cls_acc, acc_ot


def update_eval_loader(train_data, transform, args, index_x, labels, drop_last=False):

    eval_train_dataset = CustomTensorDataset(train_data[index_x], labels, transform=transform)

    class_idxs = get_idxs_per_cls_list(labels, 10)

    eval_loader = DataLoader(
            eval_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=12,
        )
    
    return eval_loader

imb_factor = 0.01
noise_ratio = 0.5

if args.dataset == 'cifar10':
    loader = dataloader.cifar_dataloader('cifar10', imb_type='exp', imb_factor=imb_factor, noise_mode=args.data_split, noise_ratio=noise_ratio,\
        batch_size=args.batch_size, num_workers=12, root_dir='./data/cifar-10-batches-py', log='log.txt')
else:
    loader = dataloader.cifar_dataloader('cifar100', imb_type='exp', imb_factor=imb_factor, noise_mode=args.data_split, noise_ratio=noise_ratio,\
        batch_size=args.batch_size, num_workers=12, root_dir='./data/cifar-100-python', log='log.txt')

train_data = torch.Tensor(loader.run('warmup').dataset.train_data).cpu().numpy()
clean_labels = torch.Tensor(loader.run('warmup').dataset.clean_targets).long()
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()

train_dataset = CustomTensorDataset(train_data, noisy_labels, transform=train_transform)

args.num_classes = torch.unique(noisy_labels).size(0)

pcl = get_noisy_imbalanced_per_cls_list(noisy_labels, args.num_classes)
_clean_labels = copy.deepcopy(clean_labels).cuda()


train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
    )

test_loader = loader.run('test')

model = ModelBase(class_num=10, MODEL=args.MODEL)
model = model.cuda()

param_list_warmup = [
    {'params': model.encoder.parameters(), 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
    {'params': model.classifier.parameters(), 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
]

optimizer_warmup = torch.optim.SGD(param_list_warmup)
criterion = nn.CrossEntropyLoss(reduction='none')

best_acc = 0.0
best_per_cls = None

results = {'train_loss': [], 'test_acc@1': []}

index = np.arange(train_data.shape[0])

clean_x = []
clean_y = []
clean_y_prob = []
p_r_a = []
sample_per_cls = []
acc_per_cls = []

eval_loader = update_eval_loader(train_data, test_transform, args, index, noisy_labels[index])

best_warm_acc = 0
best_warm_model = None

for epoch in range(300):

    warmup(epoch, model, optimizer_warmup, train_loader, criterion)

    test_acc, test_acc_per_cls, _ = test(model, test_loader, criterion, epoch, args)
    print(test_acc_per_cls)
    if test_acc > best_warm_acc:
        best_warm_acc = test_acc
        best_warm_model = copy.deepcopy(model)
     
model = best_warm_model
torch.nn.init.xavier_uniform_(model.classifier.weight)

param_list = [
    {'params': model.encoder.parameters(), 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
    {'params': model.classifier.parameters(), 'lr': 1, 'momentum': 0.9, 'weight_decay': 5e-4},
]
optimizer = torch.optim.SGD(param_list)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

init_avg_prototypes = gss(model, eval_loader, 0, args)
avg_prototypes = init_avg_prototypes

for epoch in range(1, args.epochs + 1):

    train_loss, real_sample_index = train_e(model, train_loader, optimizer, criterion, scheduler, epoch, avg_prototypes, args)
    results['train_loss'].append(train_loss)

    if args.dataset == 'cifar10':
        clean_x = torch.tensor(real_sample_index[0]).long().numpy()
        clean_y = noisy_labels[clean_x]
        clean_y_prob = real_sample_index[2]
        _cly_p = clean_y_prob

    else:
        clean_x = torch.cat([torch.tensor(clean_x), torch.tensor(real_sample_index[0])], 0)
        clean_y = torch.cat([torch.tensor(clean_y), torch.tensor(real_sample_index[1])], 0)
        clean_y_prob = torch.cat([torch.tensor(clean_y_prob), torch.tensor(real_sample_index[2])], 0)

        _, unique_indices = np.unique(clean_x.numpy(), return_index=True)

        clean_x = clean_x[unique_indices].long().numpy()
        clean_y = clean_y[unique_indices].long()
        clean_y_prob = clean_y_prob[unique_indices]
        _cly_p = clean_y_prob.cpu()

    _cl = _clean_labels[clean_x].cpu()
    _cly = clean_y.cpu()

    try:
        _auc = roc_auc_score(_cl, _cly_p, average='macro', multi_class='ovr')
    except:
        _auc = 0.0

    _precision = precision_score(_cly, _cl, average='macro')
    _recall = recall_score(_cly, _cl, average='macro')
    _acc = accuracy_score(_cly, _cl)

    print("Precision: {:.4f} Recall: {:.4f} Acc: {:.4f} AUC: {:.4f}".format(_precision, _recall, _acc, _auc))
    p_r_a.append([_precision, _recall, _acc, _auc])

    eval_loader = update_eval_loader(train_data, train_transform, args, clean_x, clean_y, True)

    print("Clean Samples: {}".format(clean_x.shape[0]))
    print("Sample Per Class: {}".format(get_noisy_imbalanced_per_cls_list(clean_y, 10)))

    kk_array = torch.zeros(10)
    for cx in clean_x:
        if noisy_labels[cx] == _clean_labels[cx]:
            kk_array[noisy_labels[cx]] += 1
    
    # print(kk_array)
    acc_per_cls_details = ', '.join('{:.4f}'.format(aaa/bbb) for aaa, bbb in zip(kk_array, get_noisy_imbalanced_per_cls_list(clean_y, args.num_classes)))
    print(acc_per_cls_details)
    acc_per_cls.append(acc_per_cls_details)

    criterion_fc = nn.CrossEntropyLoss(reduction='none')    
    train_fc(model, eval_loader, optimizer, criterion_fc, epoch, args)

    tmp_avg_prototypes = gss(model, eval_loader, 0, args)
    
    if args.dataset == 'cifar10':
        lam = 1
    else:
        if args.data_split == 'imbalance':
            lam = 0.9
        else:
            lam = 1

    avg_prototypes = lam * init_avg_prototypes + (1 - lam) * tmp_avg_prototypes
    avg_prototypes = tmp_avg_prototypes

    #evaluate the model    
    test_acc, test_acc_per_cls, test_acc_ot = test(model, test_loader, criterion, epoch, args, avg_prototypes)
    print(test_acc_per_cls)

    results['test_acc@1'].append(test_acc)
    if test_acc > best_acc:
        best_acc = test_acc
        best_per_cls = test_acc_per_cls
        best_ot_acc = test_acc_ot
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),})
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    # save model
    
    print()


print('Overall Acc: {}'.format(best_acc))
print('IF: {}'.format(imb_factor))
print('NR: {}'.format(noise_ratio))