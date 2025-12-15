"""
@inproceedings{gao2024consistent,
  title={Consistent prompting for rehearsal-free continual learning},
  author={Gao, Zhanxin and Cen, Jun and Chang, Xiaobin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28463--28473},
  year={2024}
}
基于LibContinual框架的CPrompt算法实现
231880101 孙俊晖
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from core.model.finetune import Finetune

class CPrompt_Net(nn.Module):
    def __init__(self, backbone, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.device = kwargs['device'][0] if isinstance(kwargs['device'], list) else kwargs['device']
        self.dataset_name = kwargs.get("dataset", "cifar100")
        self.init_cls_num = kwargs.get("init_cls_num", 10)
        self.inc_cls_num = kwargs.get("inc_cls_num", 10)
        self.feat_dim = kwargs.get("feat_dim", 768) 
        
        # 初始化图像编码器 - 使用ViT作为骨干网络
        self.image_encoder = backbone
        
        # 初始化分类器列表
        self.clas_w = nn.ModuleList()
        self.clas_w.append(nn.Linear(self.feat_dim, self.init_cls_num))
        
        # 初始化提示层 
        self.ts_prompts_1 = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.feat_dim))
        ])
        self.ts_prompts_2 = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.feat_dim))
        ])
        
        # 初始化键
        self.keys = nn.Parameter(torch.randn(self.init_cls_num, self.feat_dim))
        
    def update_fc(self, total_classes, cur_task_nbclasses):
        # 更新分类器
        if len(self.clas_w) == 0:
            self.clas_w.append(nn.Linear(self.feat_dim, total_classes))
        else:
            old_fc = self.clas_w[-1]
            new_fc = nn.Linear(self.feat_dim, total_classes)
            if total_classes > cur_task_nbclasses:
                new_fc.weight.data[:total_classes - cur_task_nbclasses] = old_fc.weight.data
                new_fc.bias.data[:total_classes - cur_task_nbclasses] = old_fc.bias.data
            self.clas_w.append(new_fc)
        
        # 更新提示层
        self.ts_prompts_1.append(nn.Parameter(torch.randn(1, self.feat_dim)))
        self.ts_prompts_2.append(nn.Parameter(torch.randn(1, self.feat_dim)))
        
        # 更新键
        new_keys = nn.Parameter(torch.randn(cur_task_nbclasses, self.feat_dim).to(self.device))
        if len(self.keys) == 0:
            self.keys = new_keys
        else:
            self.keys = nn.Parameter(torch.cat([self.keys, new_keys], dim=0))
    
    def aux_forward(self, x):
        features = self.image_encoder(x)
        # 确保特征是二维的
        if features.dim() > 2:
            features = features.mean(dim=[2, 3])
        logits = self.clas_w[-1](features)
        return logits, features
    
    def forward(self, x, gen_p, train=False):
        features = self.image_encoder(x)
        # 确保特征是二维的 
        if features.dim() > 2:
            features = features.mean(dim=[2, 3])
            
        # 应用提示
        P1, P2 = gen_p
        features = features + P1.squeeze(1) + P2.squeeze(1)
        logits = self.clas_w[-1](features)
        return logits
    
    def fix_branch_layer(self):
        # 冻结当前任务的参数
        for param in self.clas_w[-1].parameters():
            param.requires_grad = False
        # 冻结当前任务的提示参数
        self.ts_prompts_1[-1].requires_grad = False
        self.ts_prompts_2[-1].requires_grad = False

class CPrompt(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.device = kwargs['device'][0] if isinstance(kwargs['device'], list) else kwargs['device']

        # 任务状态管理
        self.cur_task = 0
        self.known_classes = 0
        self.total_classes = 0

        # 数据集与超参数配置
        self.dataset_name = kwargs.get("dataset", "cifar100")
        self.init_cls_num = kwargs.get("init_cls_num", 10)
        self.inc_cls_num = kwargs.get("inc_cls_num", 10)
        self.margin = kwargs.get("margin", 0.05)
        self.tau = kwargs.get("tau", 1.02)
        self.alpha = kwargs.get("alpha", 1.0)

        # 初始化网络
        self.network = CPrompt_Net(self.backbone, kwargs)
        self.network.to(self.device)

        # 记录评估指标
        self.acc = []
        self.faa_accuracy_table = []

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.cur_task = task_idx
        # 计算当前任务的类别范围
        if task_idx == 0:
            self.total_classes = self.init_cls_num
        else:
            self.total_classes += self.inc_cls_num

        # 更新网络分类器和提示
        cur_task_nbclasses = self.inc_cls_num if task_idx > 0 else self.init_cls_num
        self.network.update_fc(self.total_classes, cur_task_nbclasses)
        self.network.to(self.device)

    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        # 计算新类别标签（相对于当前任务）
        new_targets = y - self.known_classes

        # 辅助分类器损失
        logits, features = self.network.aux_forward(x)
        loss_aux = F.cross_entropy(logits, new_targets)
        loss = loss_aux

        # 分类器一致性损失（CCL）
        if self.cur_task > 0:
            for k in range(self.cur_task):
                old_logit = self.network.clas_w[k](features)
                cur_logit = self.network.clas_w[self.cur_task](features)

                # 获取当前任务和历史任务的类别数
                old_classes = self.init_cls_num if k == 0 else self.init_cls_num + k * self.inc_cls_num
                cur_classes = self.init_cls_num + self.cur_task * self.inc_cls_num

                # 只在旧类别范围内计算一致性损失
                old_logit_subset = old_logit
                cur_logit_subset = cur_logit[:, :old_classes]

                # 判断是否需要应用温度系数
                bool_mask = (torch.max(cur_logit_subset, dim=1)[0] > 
                        torch.max(old_logit_subset, dim=1)[0] + self.margin)
                t = torch.ones_like(bool_mask, dtype=torch.float32).to(self.device)
                t[~bool_mask] = self.tau
                t = t.unsqueeze(1).repeat(1, old_classes)  

                # 平滑正则化
                ground = F.softmax(old_logit / t, dim=1).detach()
                loss_ccl = -torch.sum(ground * torch.log(F.softmax(old_logit, dim=1)), dim=1).mean()
                loss += self.alpha * loss_ccl / self.cur_task

        # 多键机制损失（MK）
        with torch.no_grad():
            features = self.network.image_encoder(x)
            if features.dim() == 3:
                x_querry = features[:, 0, :]  # [batch_size, feature_dim]
            else:
                x_querry = features
        K = self.network.keys

        # 截取当前任务的键
        s = self.cur_task * self.inc_cls_num
        f = (self.cur_task + 1) * self.inc_cls_num
        if self.cur_task == 0:
            K = K[s:f]
        else:
            K = torch.cat([K[:s].detach(), K[s:f]], dim=0)

        # 余弦相似度计算
        n_K = F.normalize(K, dim=1)
        q = F.normalize(x_querry, dim=1)
        mk = torch.einsum('bd,kd->bk', q, n_K)
        loss_mk = F.cross_entropy(mk, y)
        loss += loss_mk

        # 生成提示损失（PCL）
        gen_p = []
        # 随机选择历史提示
        m = torch.randint(0, self.cur_task + 1, (x.size(0), 1))
        # 生成第一层提示
        P1 = torch.cat([self.network.ts_prompts_1[j].unsqueeze(0) for j in m], dim=0)
        gen_p.append(P1)
        # 生成第二层提示
        P2 = torch.cat([self.network.ts_prompts_2[j].unsqueeze(0) for j in m], dim=0)
        gen_p.append(P2)

        # 生成提示的分类损失
        out_gen = self.network(x, gen_p, train=True)
        loss_ce = F.cross_entropy(out_gen, new_targets)
        loss += loss_ce

        # 计算准确率
        _, preds = torch.max(logits, dim=1)
        acc = torch.sum(preds.eq(new_targets)).item() / x.size(0)

        return preds, acc, loss

    def inference(self, data):
        """推理过程处理"""
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        # 生成提示
        gen_p = []
        with torch.no_grad():
            features = self.network.image_encoder(x)
            if features.dim() == 3:
                x_querry = features[:, 0, :]  # [batch_size, feature_dim]
            else:
                x_querry = features

        # 多键机制选择提示
        K = self.network.keys
        f = (self.cur_task + 1) * self.inc_cls_num
        K = K[:f]
        n_K = F.normalize(K, dim=1)
        q = F.normalize(x_querry, dim=1)
        mk = torch.einsum('bd,kd->bk', q, n_K)
        m = torch.max(mk, dim=1, keepdim=True)[1] // self.inc_cls_num

        # 选择对应的提示
        P1 = torch.cat([self.network.ts_prompts_1[j].detach().unsqueeze(0) for j in m], dim=0)
        gen_p.append(P1)
        P2 = torch.cat([self.network.ts_prompts_2[j].detach().unsqueeze(0) for j in m], dim=0)
        gen_p.append(P2)

        # 推理预测
        with torch.no_grad():
            out_logits = self.network(x, gen_p, train=False)

        preds = torch.argmax(out_logits, dim=1)
        acc = torch.sum(preds.eq(y)).item() / x.size(0)

        return preds, acc

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        self.known_classes = self.total_classes
        self.network.fix_branch_layer()  

    def get_parameters(self, config):
        return filter(lambda p: p.requires_grad, self.network.parameters())