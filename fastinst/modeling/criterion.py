# Modified from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
FastInst criterion.
"""

import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size
from torch import nn

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 # TODO 添加 alpha_weight 参数，考虑修改成更加容易理解的 alpha_weight_start
                 alpha_weight,
                 ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # TODO: empty_weight后期删掉，暂时不删也不影响
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # TODO 添加 样本数量统计参数 cum_samples 相关的初始化处理逻辑 相关日志输出
        # 全 0
        # self.register_buffer('cum_samples',torch.zeros(self.num_classes + 1, dtype=torch.float))
        # 全 1
        cum_samples= torch.ones(self.num_classes + 1)
        # cum_samples[-1] = self.eos_coef
        self.register_buffer('cum_samples', cum_samples, persistent=True)  # 创建一个缓冲区来存储每个类别的累积样本
        # 全 1，并且不包含背景类别
        # self.register_buffer('cum_samples', torch.ones(self.num_classes, dtype=torch.float))
        self.alpha = alpha_weight

        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Set alpha_weight---alpha_weight_start = '{self.alpha}'")
        self._logger.info(f"Set register_buffer---cum_samples:  \n '{self.cum_samples}'") # cum_samples: torch.Size([81,]):
        # self._logger.info(f"Set register_buffer---empty_weight: \n '{self.empty_weight}'") # empty_weight: torch.Size([81,]):

    # TODO: loss_labels传递reweight形参
    def loss_labels(self, outputs, targets, indices, num_masks, reweight):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o
        # TODO: 改变原始的loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight) 以接收传递的reweight实参
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reweight)
        losses = {"loss_ce": loss_ce}
        return losses

    # TODO: loss_masks传递reweight形参
    def loss_masks(self, outputs, targets, indices, num_masks, reweight):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_proposals(self, output_proposals, targets, indices):
        """
        实例激活损失:计算实例激活损失，即预测的提议框与真实目标之间的交叉熵损失。
        Args:
            output_proposals:
            targets:
            indices:

        Returns: losses

        """
        assert "proposal_cls_logits" in output_proposals

        proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
        # 获取提议类别 logits:从提议输出中获取类别 logits 并将其展平。
        # 提议框类别logits通常是三维张量，形状为(batch_size, num_classes, proposal_height, proposal_width)。
        # 通过.flatten(2)操作将最后两个维度（高度和宽度）扁平化，
        # 这样就得到了形状为(batch_size, num_classes, proposal_count)的张量，其中proposal_count是所有提议框的总数。
        proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw
        # 初始化一个目标类别张量（target_classes），大小等于提议类别logits的第三维展开后的大小，
        # 全部填充为背景类别的索引（这里是背景类别的索引通常是self.num_classes，因为类别索引通常从0开始计数）。

        # 这个张量将用于存储每个提议框对应的真实类别标签。
        # 创建目标类别张量:创建一个全0的目标类别张量，其形状为提议类别 logits 的批次大小和提议区域数量的乘积。
        target_classes = self.num_classes * torch.ones([proposal_cls_logits.shape[0],
                                                        proposal_size[0] * proposal_size[1]],
                                                       device=proposal_cls_logits.device)
        target_classes = target_classes.to(torch.int64)
        # 更新目标类别张量:使用匹配索引更新目标类别张量。
        # 使用indices，它包含了提议框与真实目标之间的匹配关系，遍历每个批次的匹配结果，
        # 从真实目标的标签中提取出与提议框匹配的类别标签，并拼接成一个完整的张量target_classes_o。

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # 使用内部 _get_src_permutation_idx 函数获得提议框与真实目标匹配后的正确顺序索引（idx），
        # 用以将真实类别标签按照匹配关系填充到目标类别张量中。
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o
        # 计算提议损失,计算提议类别 logits 和目标类别之间的交叉熵损失，忽略索引为-1的元素。
        # 使用交叉熵损失函数（F.cross_entropy）计算提议框类别logits与实际类别标签之间的损失。
        # 由于存在未匹配的提议框（可以认为它们没有对应的真实目标），所以设置ignore_index=-1来忽略这些框的损失计算。
        # 计算损失:使用交叉熵损失函数（F.cross_entropy）计算提议框类别logits与实际类别标签之间的损失。
        loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
        losses = {"loss_proposal": loss_proposal}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # TODO: get_loss传递形参reweight
    def get_loss(self, loss, outputs, targets, indices, num_masks, reweight):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, reweight)  # self.loss_labels(outputs, targets, indices, num_masks)
        # return loss_map[loss](outputs, targets, indices, num_masks) # 原

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute proposal loss
        proposal_loss_dict = {}
        if outputs.get("proposal_cls_logits") is not None:
            output_proposals = {"proposal_cls_logits": outputs.pop("proposal_cls_logits")}
            indices = self.matcher(output_proposals, targets)
            proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)

        # Compute the main output loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if outputs_without_aux.get("pred_matching_indices") is not None:
            indices = outputs_without_aux["pred_matching_indices"]
        else:
            indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # TODO: forward添加计算每个类别的累计样本数 cum_samples、类别出现频率class_frequency、重加权权重reweight
        ################# 计算每个类别的样本数##############
        assert "pred_logits" in outputs   # pred_logits: torch.Size([4, 100, 81])
        # 从输出中获取预测的类别 logits 并转换为浮点数。
        src_logits = outputs["pred_logits"].float()  # src_logits:torch.Size([4, 100, 81])
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        unique_target_classes = target_classes.unique()
        for u_l in unique_target_classes:
            if u_l == self.num_classes:  # 80
                continue  # 跳过背景类的累计
                # pass  # 保留背景类的累计
                # self.cum_samples[u_l] += 20.0  # 保留背景类的累计, 但是每次只累计2背景样本数
                # self.cum_samples[u_l] += 1  # 保留背景类的累计, 但是每次只累计1背景样本数
            else:
                # 计算匹配索引中目标类别为 u_l 的样本个数
                inds = target_classes == u_l.item()  # inds: torch.Size([1,100])
                self.cum_samples[u_l] += inds.sum().float()  # torch.Size([81,])

        # total_samples = self.cum_samples.sum() - self.num_classes
        # total_samples = self.cum_samples.sum()
        total_samples = self.cum_samples[:-1].sum()
        # class_frequency = self.cum_samples / total_samples  # 包括背景类的样本比例
        class_frequency = self.cum_samples / total_samples  # 不包括背景类的样本比例，但也参与计算
        # 计算每个类别的样本权重
        reweight = self.alpha * (1 - class_frequency)**2
        # reweight = self.alpha * (1 - class_frequency ** 0.5)
        # ###########################################
        # self._logger.info("current total_samples:{} ".format(total_samples))
        # self._logger.info("current unique_target_classes:{} ".format(unique_target_classes))
        # self._logger.info("current cum_samples:\n {}".format(self.cum_samples))
        # self._logger.info("current cum_samples[0]:{}".format(self.cum_samples[0]))
        # self._logger.info("current cum_samples[1]:{}".format(self.cum_samples[1]))
        # self._logger.info("current cum_samples[2]:{}".format(self.cum_samples[2]))
        # self._logger.info("current cum_samples[3]:{}".format(self.cum_samples[3]))
        # self._logger.info("current cum_samples[80]:{}".format(self.cum_samples[80]))
        # self._logger.info("class_frequency ：\n {}".format(class_frequency))
        # self._logger.info("current reweight = self.alpha * (1 - class_frequency ** 0.5) to format:\n {} \n".format(reweight))
        # reweight[80] = 0.1
        reweight[80] = 1.0
        # self._logger.info("current cum_samples[80]:{}".format(self.cum_samples[80]))
        # self._logger.info(
        #     "current reweight = self.alpha * (1 - class_frequency ** 0.5) to format:\n {} \n".format(reweight))
        # ###########################################

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
            # TODO: 将计算出的重加权权重reweight 添加到损失函数中
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, reweight))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if aux_outputs.get("pred_matching_indices") is not None:
                    indices = aux_outputs["pred_matching_indices"]
                else:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    # TODO: reweight 参数传递
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, reweight)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses.update(proposal_loss_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
