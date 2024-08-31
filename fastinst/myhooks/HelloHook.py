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
        if self.trainer.iter % 500 == 0:
            # if self.trainer.iter % 100 == 0:
            self._logger = logging.getLogger(__name__)
            self._logger.info("HelloHook Logger at iteration: {}".format(self.trainer.iter))
            self._logger.info("trainer max iteration {}".format(self.trainer.max_iter))
            self._logger.info("current training at iteration {}".format(self.trainer.iter))
            self._logger.info("current cum_samples = {}". format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.cum_samples))
            # _total_samples = self.cum_samples[:-1].sum() # self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.cum_samples
            _total_samples = self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.cum_samples[:-1].sum()
            self._logger.info("current total_samples = {}". format(_total_samples))
            _class_frequency = self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.cum_samples / _total_samples  # 不包括背景类的样本比例，但也参与计算
            self._logger.info("current class_frequency = {}". format(_class_frequency))
            self._logger.info("current alpha = {}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha))
            # self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha = 
            # 计算每个类别的样本权重
            # current_reweight = self.alpha * (1 - _class_frequency)**2
            current_reweight = self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha * (1 - _class_frequency)**2
            self._logger.info("current reweight = {}". format(current_reweight))
            
           
class ALPHA_DramHook(HookBase):
    """
    ALPHA权重修改Hook：
    criterion.alpha += self.val_change
    """
    def __init__(self, val_start, val_end,num_change=50):
        self._logger = logging.getLogger(__name__)
        self._num_change = num_change
        self._start = val_start
        self._end = val_end
        self.val_change = (self._end - self._start) / self._num_change
        self._logger.info("ALPHA_DramHook enabled : val_start = {}, val_end = {}, num_change = {}, val_change = {}".format(self._start, self._end, self._num_change, self.val_change))

    # def before_train(self):
    #     self._logger.info("========before_train===========")
    #     self._logger.info("trainer max iteration {}".format(self.trainer.max_iter))
    #     self._logger.info("current training at iteration {}".format(self.trainer.iter))
    #     self._logger.info("current empty_weight[-1] = {}". format(
    #         self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))

    # def after_train(self):
    #     self._logger.info("========after_train===========")
    #     self._logger.info("trainer max iteration {}".format(self.trainer.max_iter))
    #     self._logger.info("current training at iteration {}".format(self.trainer.iter))
    #     self._logger.info("current empty_weight[-1] = {}".format(
    #         self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))

    def after_step(self):
        period_change = self.trainer.max_iter / self._num_change
        if self.trainer.iter % 200 == 0:
            self._logger.info("trainer max iteration {}, current training at iteration {}".format(self.trainer.max_iter,self.trainer.iter))
            self._logger.info("current cum_samples= {}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.cum_samples))
            class_frequency = self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.cum_samples / self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.cum_samples[:-1].sum()
            self._logger.info("current class_frequency = {}".format(class_frequency))
            self._logger.info("Dynamic reweight = self.alpha * (1 - class_frequency)**2, backgroud_weight = 1") 
            self._logger.info("current alpha = :{}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha)) 
            self._logger.info("current reweight = {}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha*(1 - class_frequency)**2))
            
        if self.trainer.iter != 0 and self.trainer.iter % period_change == 0:
            self._logger.info("after_step--it is time to change alpha---current iter: {}, change value: {}".format(self.trainer.iter, self.val_change))  
            self._logger.info("self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha:{}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha)) 
            # self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1] += self.val_change
            self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha += self.val_change
            self._logger.info("ATTENTION: current training at iteration {}, alpha change to  {}".format(self.trainer.iter, self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.alpha))



class DramHook(HookBase):
    def __init__(self, num_change=50, val_start=0.05, val_end=0.15, type='increase'):
        self._logger = logging.getLogger(__name__)
        self._num_change = num_change
        self._start = val_start
        self._end = val_end
        self._type = type
        self.val_change = (self._end - self._start) / self._num_change
        self._logger.info("DramHook enabled : val_start = {}, val_start = {}, num_change = {}, val_change = {}, type = {}".format(self._start, self._end, self._num_change, self.val_change, self._type))

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
        period_change = self.trainer.max_iter / self._num_change
        # self._logger.info("after_step-----empty_weight[-1]: {}".format(self.trainer.model.criterion.empty_weight[-1]))
        if self.trainer.iter != 0 and self.trainer.iter % period_change == 0:
        # if self.trainer.iter != 0 and self.trainer.iter % self.period_change == 0:
            if self._type == 'increase':
                # val_change = (self._end - self._start) / self._num_change
                # self.val_change = (self._end - self._start) / self._num_change
                self._logger.info("after_step--it is time to change weight---current iter: {}, increase value: {}".format(self.trainer.iter, self.val_change))  # 0.0007999999999999999
                self._logger.info("self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]:{}".format(self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1])) # 0.10000000149011612
                self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1] += self.val_change
                self._logger.info("ATTENTION: current training at iteration {}, no_object_weight change to  {}".format(self.trainer.iter, self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))
            elif self._type == 'decrease':
                # val_change = (self._start - self._end) / self._num_change
                # self.trainer.model.criterion.empty_weight[-1] -= self.val_change
                self._logger.info(
                    "after_step--it is time to change weight---current iter: {}, decrease value: {}".format(
                        self.trainer.iter, self.val_change))  # 0.0007999999999999999
                self._logger.info(
                    "self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]:{}".format(
                        self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[
                            -1]))  # 0.10000000149011612
                self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[
                    -1] -= self.val_change
                self._logger.info("ATTENTION: current training at iteration {}, no_object_weight change to  {}".format(
                    self.trainer.iter,
                    self.trainer._trainer.model._modules['module'].sem_seg_head.predictor.criterion.empty_weight[-1]))
            else:
                self._logger.info("Unsupported DramHook type: {}".format(self._type))
