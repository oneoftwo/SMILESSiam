import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from util.train import evaluate as EVALUATE


def process_siam(model, data_loader, optimizer=None, args=None):
    model.cuda()
    if not optimizer == None:
        model.train()
    else:
        model.eval()

    criterion_1 = nn.CosineSimilarity()
    criterion_2 = nn.MSELoss()
    total_loss, total_data, total_loss_siam, total_loss_pp = 0, 0, 0, 0
    start_time = time.time()

    result_dict = {}
    all_z = []
    for batch in data_loader:
        l1, l2, s1, s2 = batch['length_1'].long(), batch['length_2'].long(), \
                batch['seq_1'].long(), batch['seq_2'].long()
        sample = model(s1.cuda(), l1, s2.cuda(), l2)
        p1, p2, z1, z2 = sample['p1'], sample['p2'], sample['z1'].detach(), sample['z2'].detach()
        
        if args.use_pp_prediction:
            pp1, pp2 = sample['pp1'], sample['pp2']
        
        loss_siam = -(criterion_1(p1, z2).sum() + criterion_1(p2, z1).sum()) * 0.5
        loss_siam = loss_siam.cpu()
        total_loss_siam += loss_siam.item()

        if args.use_pp_prediction:
            loss_pp = (criterion_2(pp1, batch['pp'].cuda()) + criterion_2(pp2, batch['pp'].cuda())) * 0.5
            loss_pp = loss_pp.cpu()
            loss = loss_siam + loss_pp * args.pp_loss_ratio
            total_loss_pp += loss_pp.item()
        else:
            loss = loss_siam

        if not optimizer == None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_data += z1.size(0)
        all_z.append(z1) # add only z1 
    
    all_z = torch.cat(all_z, dim=0)
    std = calc_latent_std(all_z)

    result_dict['std'] = np.mean(std)
    result_dict['loss_siam'] = total_loss_siam / total_data 

    if args.use_pp_prediction:
        result_dict['loss_pp'] = total_loss_pp / total_data 
    else:
        result_dict['loss_pp'] = 0

    result_dict['time'] = time.time() - start_time
    return model, result_dict


def process_clf(model, data_loader, optimizer=None):
    model.cuda()
    if not optimizer == None:
        model.train()
    else:
        model.eval()

    criterion = nn.BCELoss(reduction='sum')
    total_loss, total_data = 0, 0  
    total_pred_p, total_true_label = [], []
    start_time = time.time()
    
    result_dict = {}
    for batch in data_loader:
        l, s = batch['length_1'].long(), batch['seq_1'].long()
        true_target = batch['target'].long().cuda()
        pred_p= torch.sigmoid(model(s.cuda(), l))
        loss = criterion(pred_p, true_target.float())

        if not optimizer == None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_data += pred_p.size(0)
        total_pred_p += pred_p.tolist()
        total_true_label += true_target.tolist()

    total_pred_p = np.array(total_pred_p)
    total_true_label = np.array(total_true_label)
    total_pred_label = np.round(total_pred_p)
    
    result_dict = {}
    result_dict['accuracy'] = EVALUATE.calc_accuracy(total_pred_label, total_true_label)
    result_dict['precision'] = EVALUATE.calc_precision(total_pred_label, total_true_label)
    result_dict['recall'] = EVALUATE.calc_recall(total_pred_label, total_true_label)
    result_dict['auc_roc'] = EVALUATE.calc_roc_auc(total_pred_p, total_true_label)
    result_dict['auc_prc'] = EVALUATE.calc_prc_auc(total_pred_p, total_true_label)
    result_dict['loss'] = total_loss / total_data 
    result_dict['time'] = time.time() - start_time

    return model, result_dict 


def process_clf_validation_smiles_enumerate(model, data_loader, n_trial=100):
    # data_loader should not be shuffled !!!
    model.cuda()
    model.eval()

    criterion = nn.BCELoss(reduction='sum')
    total_loss, total_data = 0, 0  
    start_time = time.time()
    
    result_dict = {}
    total_pred_p_score = []
    for _ in range(n_trial):
        enm_pred_p_score, enm_true_label = [], []
        for batch in data_loader:
            l, s = batch['length_1'].long(), batch['seq_1'].long()
            true_target = batch['target'].long().cuda()
            pred_p_score = model(s.cuda(), l)
            loss = criterion(torch.sigmoid(pred_p_score), true_target.float())
            
            total_loss += loss.item() # float
            total_data += pred_p_score.size(0) # int
            
            enm_pred_p_score.append(pred_p_score)
            enm_true_label.append(true_target)

        enm_pred_p_score = torch.cat(enm_pred_p_score, dim=0)
        enm_true_label = torch.cat(enm_true_label, dim=0)
        total_pred_p_score.append(enm_pred_p_score)

    total_pred_p_score = torch.stack(total_pred_p_score, dim=0) # [n_trail, n_data]
    avg_pred_p_score = total_pred_p_score.mean(dim=0) # [n_data]
    avg_pred_p = torch.sigmoid(avg_pred_p_score) # apply sigmoid after
    total_true_label = enm_true_label
    
    total_true_label = total_true_label.cpu().detach().numpy()
    avg_pred_p = avg_pred_p.cpu().detach().numpy()
    avg_pred_label = np.round(avg_pred_p)
    
    total_pred_label, total_pred_p, total_true_label = \
            avg_pred_label, avg_pred_p, total_true_label
    result_dict = {}
    result_dict['accuracy'] = EVALUATE.calc_accuracy(total_pred_label, total_true_label)
    result_dict['precision'] = EVALUATE.calc_precision(total_pred_label, total_true_label)
    result_dict['recall'] = EVALUATE.calc_recall(total_pred_label, total_true_label)
    result_dict['auc_roc'] = EVALUATE.calc_roc_auc(total_pred_p, total_true_label)
    result_dict['auc_prc'] = EVALUATE.calc_prc_auc(total_pred_p, total_true_label)
    result_dict['loss'] = total_loss / total_data
    result_dict['time'] = time.time() - start_time

    return model, result_dict 


def calc_latent_std(x): # x[bs hd]
    x = x.cpu().detach().numpy()
    std = np.std(x, axis=1)
    return std 
    

if __name__ == '__main__':
    x = torch.rand(16, 1000)
    y = calc_latent_std(x)
    print(y)

