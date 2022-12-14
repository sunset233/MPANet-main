from layers.module.CBAM import *
from layers.module.NonLocal import Non_local
from models.resnet import resnet50
from utils.calc_acc import calc_acc
from hrnet.getAttributeFeature import *
from layers.loss.triplet_loss import *
from layers import CenterTripletLoss
from layers import CenterLoss
from models.pos_embedding import *

class ASN(nn.Module):
    def __init__(self, pool_dim):
        super(ASN, self).__init__()
        self.IN = nn.InstanceNorm2d(pool_dim)
        self.BN = nn.BatchNorm2d(pool_dim)
        self.cbam = cbam(pool_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(pool_dim, pool_dim, kernel_size=1)

    def forward(self, x):
        x_in = self.IN(x)
        y1 = self.cbam(x - x_in) + x_in
        x_bn = self.BN(x)
        y2 = self.cbam(x_bn)
        y = y1 + 2 * y2
        return y

class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        x = x * mask + self.IN(x) * (1 - mask)
        return x

class backbone(nn.Module):
    def __init__(self, arch='resnet50'):
        super(backbone, self).__init__()
        self.base = resnet50(pretrained=True)
        self.MAM1 = MAM(1024)
        self.ASN2 = ASN(2048)
        layers = [3, 4, 6, 3]
        non_layers = [0, 2, 3, 0]
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        # self.transform = MHSA(1024, width=24, height=8, heads=4)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.base.layer3)):
            x = self.base.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # x = self.MAM1(x)
        # x = self.transform(x)
        x = self.base.layer4(x)
        x = self.ASN2(x)
        return x

class Baseline(nn.Module):
    def __init__(self, model_attr=None, num_classes=None, drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning
        self.backbone = backbone()
        self.modedl_attr = model_attr
        self.base_dim = 2048
        self.dim = 0
        self.part_num = kwargs.get('num_parts', 0)

        if pattern_attention:
            self.part_head = backbone()
            self.base_dim = 2048
            self.dim = 2048
            self.part_num = kwargs.get('num_parts', 3)
            self.spatial_attention = nn.Conv2d(self.base_dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)
            torch.nn.init.constant_(self.spatial_attention.bias, 0.0)
            self.activation = nn.Sigmoid()
            self.weight_sep = kwargs.get('weight_sep', 0.1)


        if mutual_learning:
            self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

            self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.visible_classifier_.weight.requires_grad_(False)
            self.visible_classifier_.weight.data = self.visible_classifier.weight.data

            self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.infrared_classifier_.weight.requires_grad_(False)
            self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

            self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
            self.weight_sid = kwargs.get('weight_sid', 0.5)
            self.weight_KL = kwargs.get('weight_KL', 2.0)
            self.update_rate = kwargs.get('update_rate', 0.2)
            self.update_rate_ = self.update_rate

        print("output feat length:{}".format(self.base_dim + self.dim * self.part_num))
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)
        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.center_cluster = kwargs.get('center_cluster', False)
        self.center_loss = kwargs.get('center', False)
        self.margin = kwargs.get('margin', 0.3)

        if self.classification:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            # self.classifier.apply()
        if self.mutual_learning or self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.triplet:
            self.triplet_loss = TripletLoss(margin=self.margin)
            # self.triplet_loss = TripletLoss_WRT()
        if self.center_cluster:
            k_size = kwargs.get('k_size', 8)
            self.center_cluster_loss = CenterTripletLoss(k_size=k_size, margin=self.margin)
        if self.center_loss:
            self.center_loss = CenterLoss(num_classes, self.base_dim + self.dim * self.part_num)

    def forward(self, inputs, labels=None, **kwargs):
        loss_tri = 0
        loss_center = 0
        modality_logits = None
        modality_feat = None
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)

        # CNN
        global_feat = self.backbone(inputs)
        b, c, w, h = global_feat.shape
        feats = []
        feat_mask = []

        if self.pattern_attention:
            mask, mask_1, mask_2, mask_3 = pred_imgs(inputs, self.modedl_attr)
            # mask_global feature process
            mask = inputs * mask.unsqueeze(1).repeat(1,3,1,1)
            feat = self.part_head(mask)
            feat = F.avg_pool2d(feat, feat.size()[2:])
            feat_mask = feat.view(feat.size(0), -1)

            # mask_head feature process
            feat1 = inputs * mask_1.unsqueeze(1).repeat(1,3,1,1)
            feat1 = self.part_head(feat1)
            feat1 = F.avg_pool2d(feat1, feat1.size()[2:])
            feat1 = feat1.view(feat1.size(0), -1)
            feats.append(feat1)

            # mask_upper feature process
            feat2 = inputs * mask_2.unsqueeze(1).repeat(1,3,1,1)
            feat2 = self.part_head(feat2)
            feat2 = F.avg_pool2d(feat2, feat2.size()[2:])
            feat2 = feat2.view(feat2.size(0), -1)
            feats.append(feat2)

            # mask_downer feature process
            feat3 = inputs * mask_3.unsqueeze(1).repeat(1,3,1,1)
            feat3 = self.part_head(feat3)
            feat3 = F.avg_pool2d(feat3, feat3.size()[2:])
            feat3 = feat3.view(feat3.size(0), -1)
            feats.append(feat3)

            global_feat = F.avg_pool2d(global_feat, global_feat.size()[2:])
            global_feat = global_feat.view(global_feat.size(0), -1)
            feats.append(global_feat)
            feats = torch.cat(feats, 1)

        else:
            feats = F.avg_pool2d(global_feat, global_feat.size()[2:])
            feats = feats.view(feats.size(0), -1)

        if not self.training:
            feats = self.bn_neck(feats)
            return feats
        else:
            return self.train_forward(global_feat, feat_mask, feats, labels, sub, **kwargs)

    def train_forward(self, global_feat, feat_mask, feats, labels, sub, **kwargs):
        epoch = kwargs.get('epoch')
        metric = {}
        if self.pattern_attention:
            global_feat = F.normalize(global_feat.view(global_feat.size(0), -1))
            feat_mask = F.normalize(feat_mask.view(feat_mask.size(0), -1))
            # loss_reg = 1 - torch.cosine_similarity(global_feat, feat_mask, dim=0).float()
            loss_reg = (global_feat - feat_mask) ** 2 * ((global_feat > 0) | (feat_mask > 0)).float()
            loss_reg = torch.abs(loss_reg).sum()
            loss = loss_reg.float() * self.weight_sep
            metric.update({'p-reg': loss_reg.data})
        else:
            loss = 0

        if self.triplet:
            triplet_loss, _ = self.triplet_loss(feats.float(), labels)
            loss += triplet_loss
            metric.update({'tri': triplet_loss.data})
            # loss_tri = self.triplet_loss(global_feat.float(), labels)
            # loss += loss_tri
            # metric.update({'tri_cross': loss_tri.data})

        if self.center_loss:
            center_loss = self.center_loss(feats.float(), labels)
            loss += center_loss
            metric.update({'cen': center_loss.data})

        if self.center_cluster:
            center_cluster_loss, _, _ = self.center_cluster_loss(feats.float(), labels)
            loss += center_cluster_loss
            metric.update({'cc': center_cluster_loss.data})

        feats = self.bn_neck(feats)

        if self.classification:
            # feat = F.dropout(feat, 0.3, True)
            # feat = F.relu(feat)
            logits = self.classifier(feats)
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'acc': calc_acc(logits.data, labels), 'ce': cls_loss.data})

        if self.mutual_learning:
            # cam_ids = kwargs.get('cam_ids')
            # sub = (cam_ids == 3) + (cam_ids == 6)

            logits_v = self.visible_classifier(feats[sub == 0])
            v_cls_loss = self.id_loss(logits_v.float(), labels[sub == 0])
            loss += v_cls_loss * self.weight_sid
            logits_i = self.infrared_classifier(feats[sub == 1])
            i_cls_loss = self.id_loss(logits_i.float(), labels[sub == 1])
            loss += i_cls_loss * self.weight_sid

            logits_m = torch.cat([logits_v, logits_i], 0).float()
            with torch.no_grad():
                self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                 + self.infrared_classifier.weight.data * self.update_rate
                self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                 + self.visible_classifier.weight.data * self.update_rate

                logits_v_ = self.infrared_classifier_(feats[sub == 0])
                logits_i_ = self.visible_classifier_(feats[sub == 1])

                logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()
            logits_m = F.softmax(logits_m, 1)
            logits_m_ = F.log_softmax(logits_m_, 1)
            mod_loss = self.KLDivLoss(logits_m_, logits_m) 

            loss += mod_loss * self.weight_KL + (v_cls_loss + i_cls_loss) * self.weight_sid
            metric.update({'ce-v': v_cls_loss.data})
            metric.update({'ce-i': i_cls_loss.data})
            metric.update({'KL': mod_loss.data})

        return loss, metric
