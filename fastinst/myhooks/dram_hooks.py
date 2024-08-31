import concurrent.futures
import logging
import numpy as np
import time
import weakref
from typing import List, Mapping, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage

from detectron2.evaluation.testing import flatten_results_dict
from detectron2.solver import LRMultiplier
from detectron2.utils.events import EventStorage, EventWriter
from detectron2.utils.file_io import PathManager

# from .train_loop import HookBase
# from detectron2.engine.train_loop import HookBase
from detectron2.engine import HookBase

__all__ = ["HookBase", "HelloHook", "DramHook"]


class HelloHook(HookBase):
    def after_step(self):
        if self.trainer.iter % 40 == 0:
            # if self.trainer.iter % 100 == 0:
            self._logger = logging.getLogger(__name__)
            self._logger.info("Hello at iteration {}".format(self.trainer.iter))
            # self._logger.info("start_iter {}".format(self.trainer.start_iter))
            # self._logger.info("max_iter {}".format(self.trainer.max_iter))
            # self._logger.info("storage {}".format(self.trainer.storage))
            # self._logger.info("model: {}".format(self.trainer.model))
            # self._logger.info("backbone: {}".format(self.trainer.model.backbone))
            # self._logger.info("predictor: {}".format(self.trainer.model.predictor)) # 错误
            # self._logger.info("criterion: {}".format(self.trainer.model.sem_seg_head.criterion))
            # self._logger.info("self.trainer.cfg--- {}".format(self.trainer.cfg))
            # self._logger.info("trainer.cfg.MODEL.WEIGHTS{} ".format(self.trainer.cfg.MODEL.WEIGHTS))
            # self._logger.info("cfg {}!".format(self.trainer.storage.))
            # logger = logging.getLogger("detectron2.trainer")
            # logger.info("detectron2.trainer ...")

            self._logger.info("================================start")
            self._logger.info(
                "trainer.model: {}".format(self.trainer.model))  # FastInst((backbone) (sem_seg_head) (criterion))
            # self._logger.info("trainer.model.backbone: {}".format(self.trainer.model.backbone))
            self._logger.info("trainer.model.sem_seg_head: {}".format(self.trainer.model.sem_seg_head))  # FastInstHead
            self._logger.info(
                "trainer.model.sem_seg_head.num_class: {}".format(self.trainer.model.sem_seg_head.num_classes))  # 80
            self._logger.info("trainer.model.sem_seg_head.pixel_decoder: {}".format(
                self.trainer.model.sem_seg_head.pixel_decoder))  # PyramidPoolingModuleFPN
            self._logger.info("trainer.model.sem_seg_head.predictor: {}".format(
                self.trainer.model.sem_seg_head.predictor))  # FastInstDecoder

            # self._logger.info("trainer.model.sem_seg_head.criterion: {}".format(self.trainer.model.sem_seg_head.criterion))  # 错误方式
            self._logger.info(
                "trainer.model.criterion: {}".format(self.trainer.model.criterion))  # Criterion SetCriterion
            self._logger.info(
                "trainer.model.criterion.weight_dict: {}".format(self.trainer.model.criterion.weight_dict))
            self._logger.info(
                "trainer.model.criterion.eos_coef: {}".format(self.trainer.model.criterion.eos_coef))  # 0.1
            self._logger.info(
                "trainer.model.criterion.losses: {}".format(self.trainer.model.criterion.losses))  # ['labels', 'masks']
            self._logger.info("trainer.model.criterion.matcher: {}".format(
                self.trainer.model.criterion.matcher))  # Matcher HungarianMatcher
            self._logger.info(
                "trainer.model.criterion.empty_weight: {}".format(self.trainer.model.criterion.empty_weight))
            self._logger.info(
                "trainer.model.criterion.empty_weight: {}".format(self.trainer.model.criterion.empty_weight[-1]))
            self.trainer.model.criterion.empty_weight[-1] += 0.001
            self._logger.info(
                "trainer.model.criterion.num_points: {}".format(self.trainer.model.criterion.num_points))  # 12544
            self._logger.info("trainer.model.criterion.oversample_ratio: {}".format(
                self.trainer.model.criterion.oversample_ratio))  # 3.0
            self._logger.info(
                "trainer.model.criterion.alpha: {}".format(self.trainer.model.criterion.alpha))  # alpha: 8.0 ++ 11.0
            self.trainer.model.criterion.alpha += 1
            self.trainer.model.criterion.eos_coef += 0.1
            self._logger.info("trainer.model.num_queries: {}".format(self.trainer.model.num_queries))  # 100
            # self._logger.info("trainer.model.input_shape: {}".format(self.trainer.model.input_shape))  # error
            self._logger.info("================================end")  # 100
            # self._logger.info("trainer.model.overlap_threshold: {}".format(self.trainer.model.overlap_threshold))  # 0.8

        # self._logger.info("trainer.model.sem_seg_head.class_embed_layers: {}".format(self.trainer.model.sem_seg_head.class_embed_layers))  # 错误方式
        # self._logger.info("trainer.model.sem_seg_head.mask_embed_layers: {}".format(self.trainer.model.sem_seg_head.mask_embed_layers))  # # 错误方式
        # FastInstHead(
        # (pixel_decoder)
        # (predictor): FastInstDecoder(
        #     (empty_query_features): Embedding(8, 256)
        #     (empty_query_pos_embed): Embedding(8, 256)
        #     (query_proposal): QueryProposal(
        #       (conv_proposal_cls_logits): Sequential(
        #         (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #         (1): ReLU(inplace=True)
        #         (2): Conv2d(256, 81, kernel_size=(1, 1), stride=(1, 1))
        #       )
        #     )
        # (transformer_query_cross_attention_layers)
        # (transformer_query_self_attention_layers)
        # (transformer_query_ffn_layers)
        # (transformer_mask_cross_attention_layers)
        # (transformer_mask_ffn_layers)
        #
        # (decoder_query_norm_layers)
        # (class_embed_layers)
        # (mask_embed_layers)
        # (mask_features_layers)
        # (criterion)
        #   )
        #     self._logger.info("model.sem_seg_head.pixel_decoder: {}".format(self.trainer.model.sem_seg_head.pixel_decoder)) # PyramidPoolingModuleFPN

        # self._logger.info("trainer.model.sem_seg_head.predictor: {}".format(self.trainer.model.sem_seg_head.predictor))  # FastInstDecoder
        #

        # self._logger.info("trainer.model.sem_seg_head.predictor.criterion: {}".format(self.trainer.model.sem_seg_head.predictor.criterion))  # Criterion SetCriterion


class DramHook(HookBase):
    """
    背景类权重修改Hook：
    criterion.empty_weight[-1] += self.val_change
    """
    def __init__(self, num_change=50, val_start=0.05, val_end=0.15):
        self._logger = logging.getLogger(__name__)
        self._num_change = num_change
        self._start = val_start
        self._end = val_end
        self._type = type
        self.val_change = (self._end - self._start) / self._num_change
        self._logger.info("DramHook enabled : val_start = {}, val_start = {}, num_change = {}, val_change = {}".format(self._start, self._end, self._num_change, self.val_change))

    # def before_step(self):
    #     self._logger.info("before_step-----empty_weight[-1]: {}".format(self.trainer.model.criterion.empty_weight[-1]))
    #     self._logger.info("before_step-----empty_weight[-1]: {}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))

    def before_train(self):
        self._logger.info("========before_train===========")
        self._logger.info("trainer max iteration {}".format(self.trainer.max_iter))
        self._logger.info("current training at iteration {}".format(self.trainer.iter))
        self._logger.info("current empty_weight[-1] = {}". format(
            self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))

    def after_train(self):
        self._logger.info("========after_train===========")
        self._logger.info("trainer max iteration {}".format(self.trainer.max_iter))
        self._logger.info("current training at iteration {}".format(self.trainer.iter))
        self._logger.info("current empty_weight[-1] = {}".format(
            self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))

    def after_step(self):
        # period_change = self.trainer.max_iter / self._num_change
        period_change = self.trainer.max_iter // self._num_change
        # self._logger.info("period_change: {}".format(period_change))

        # self._logger.info("after_step-----empty_weight[-1]: {}".format(self.trainer.model.criterion.empty_weight[-1]))
        if self.trainer.iter != 0 and self.trainer.iter % period_change == 0:
            self._logger.info("after_step--It is time to change weight---current iter: {}, increase value: {}".format(self.trainer.iter, self.val_change))  # 0.0007999999999999999
            self._logger.info("self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]:{}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))  # 0.10000000149011612
            self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1] += self.val_change
            self._logger.info("ATTENTION: current training at iteration {}, no_object_weight change to  {}".format(self.trainer.iter,self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))
            self._logger.info(
                "self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]:{}".format(
                    self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[
                        -1]))  # 0.10000000149011612

        # if self.trainer.iter != 0 and self.trainer.iter % self.period_change == 0:
        #     if self._type == 'increase':
        #         # val_change = (self._end - self._start) / self._num_change
        #         # self.val_change = (self._end - self._start) / self._num_change
        #         self._logger.info("after_step--it is time to change weight---current iter: {}, increase value: {}".format(self.trainer.iter, self.val_change))  # 0.0007999999999999999
        #         self._logger.info("self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]:{}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1])) # 0.10000000149011612
        #         self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1] += self.val_change
        #         self._logger.info("ATTENTION: current training at iteration {}, no_object_weight change to  {}".format(self.trainer.iter, self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))
        #     elif self._type == 'decrease':
        #         # val_change = (self._start - self._end) / self._num_change
        #         # self.trainer.model.criterion.empty_weight[-1] -= self.val_change
        #         self._logger.info(
        #             "after_step--it is time to change weight---current iter: {}, decrease value: {}".format(
        #                 self.trainer.iter, self.val_change))  # 0.0007999999999999999
        #         self._logger.info(
        #             "self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]:{}".format(
        #                 self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[
        #                     -1]))  # 0.10000000149011612
        #         self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[
        #             -1] -= self.val_change
        #         self._logger.info("ATTENTION: current training at iteration {}, no_object_weight change to  {}".format(
        #             self.trainer.iter,
        #             self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))
        #     else:
        #         self._logger.info("Unsupported DramHook type: {}".format(self._type))
