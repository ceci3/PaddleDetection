# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.nas.search_space.search_space_base import SearchSpaceBase
from paddleslim.nas.search_space.search_space_registry import SEARCHSPACE
from ppdet.modeling.backbones.blazenet import BlazeNet
from ppdet.modeling.architectures.blazeface import BlazeFace

@SEARCHSPACE.register
class BlazeFaceSpace(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
    #def __init__(self, input_size, output_size, block_num, block_mask, output_decoder, min_size, use_density_prior_box):
        super(BlazeFaceSpace, self).__init__(input_size, output_size, block_num, block_mask)
        self.blaze_filter_num1 = np.array([4, 8, 12, 16, 24, 32])
        self.blaze_filter_num2 = np.array([16, 24, 32, 48, 64, 72, 80])
        self.double_out_filter_num = np.array([8, 12, 16, 24, 32, 48]) 
        self.double_filter_num = np.array([16, 32, 48, 64, 72, 96, 104, 112]) # 0 mean None double blaze_filter
        #self.double_filter_num = np.array([0, 16, 32, 48, 64, 72, 96, 104, 112]) # 0 mean None double blaze_filter
        self.with_extra_blocks_num = np.array([0, 1])
        ###self.k_size = np.array([3, 5])  ##search for kernel size
        #self.blaze_filters = [[24, 24], [24, 24], [24, 48, 2], [48, 48], [48, 48]]
        #self.double_blaze_filters=[[48, 24, 96, 2], [96, 24, 96], [96, 24, 96], [96, 24, 96, 2], [96, 24, 96], [96, 24, 96]]
        #self.with_extra_blocks = False
        self.lite_edition = False

    def init_tokens(self):
        """
        """
        return [0] * 8 

    def range_table(self):
        """
        """
        return [len(self.blaze_filter_num1), len(self.blaze_filter_num2),
                len(self.double_out_filter_num), len(self.double_filter_num),
                len(self.double_out_filter_num), len(self.double_out_filter_num),
                len(self.double_out_filter_num), len(self.with_extra_blocks_num)]

    def token2arch(self, tokens=None):

        if tokens is None:
            tokens = self.init_tokens()

#        self.blaze_filters = [[self.blaze_filter_num1[tokens[0]], self.blaze_filter_num1[tokens[1]]], 
#                              [self.blaze_filter_num1[tokens[1]], self.blaze_filter_num1[tokens[2]]],
#                              [self.blaze_filter_num1[tokens[2]], self.blaze_filter_num2[tokens[3]], 2],
#                              [self.blaze_filter_num2[tokens[3]], self.blaze_filter_num2[tokens[4]]],
#                              [self.blaze_filter_num2[tokens[4]], self.blaze_filter_num2[tokens[5]]]]
        self.blaze_filters = [[self.blaze_filter_num1[tokens[0]], self.blaze_filter_num1[tokens[0]]], 
                      #        [self.blaze_filter_num1[tokens[0]], self.blaze_filter_num1[tokens[0]]],
                              [self.blaze_filter_num1[tokens[0]], self.blaze_filter_num2[tokens[1]], 2],
                       #       [self.blaze_filter_num2[tokens[1]], self.blaze_filter_num2[tokens[1]]],
                              [self.blaze_filter_num2[tokens[1]], self.blaze_filter_num2[tokens[1]]]]

        ### self.double_blaze_filters include 6 list
        #self.double_blaze_filters = [[self.blaze_filter_num2[tokens[5]], self.double_out_filter_num[tokens[6]], 
        #      self.double_filter_num[tokens[7]] if self.double_filter_num[tokens[7]] != 0 else None, 2],  ### 1
        #      [self.double_filter_num[tokens[7]] if self.double_filter_num[tokens[7]] != 0 else self.double_out_filter_num[tokens[6]],
        #      self.double_out_filter_num[tokens[8]], 
        #      self.double_filter_num[tokens[9]] if self.double_filter_num[tokens[9]] != 0 else None], ### 2
        #      [self.double_filter_num[tokens[9]] if self.double_filter_num[tokens[9]] != 0 else self.double_out_filter_num[tokens[8]],
        #      self.double_out_filter_num[tokens[10]],
        #      self.double_filter_num[tokens[11]] if self.double_filter_num[tokens[11]] != 0 else None], ### 3
        #      [self.double_filter_num[tokens[11]] if self.double_filter_num[tokens[11]] != 0 else self.double_out_filter_num[tokens[10]], 
        #      self.double_out_filter_num[tokens[12]],
        #      self.double_filter_num[tokens[13]] if self.double_filter_num[tokens[13]] != 0 else None, 2], ### 4
        #      [self.double_filter_num[tokens[13]] if self.double_filter_num[tokens[13]] != 0 else self.double_out_filter_num[tokens[12]], 
        #      self.double_out_filter_num[tokens[14]],
        #      self.double_filter_num[tokens[15]] if self.double_filter_num[tokens[15]] != 0 else None], ### 5
        #      [self.double_filter_num[tokens[15]] if self.double_filter_num[tokens[15]] != 0 else self.double_out_filter_num[tokens[14]],
        #      self.double_out_filter_num[tokens[16]],
        #      self.double_filter_num[tokens[17]] if self.double_filter_num[tokens[17]] != 0 else None], ### 6
        #      ]
        self.double_blaze_filters = [[self.blaze_filter_num2[tokens[1]], self.double_out_filter_num[tokens[2]], 
              self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else None, 2],  ### 1
              [self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else self.double_out_filter_num[tokens[2]],
              self.double_out_filter_num[tokens[4]], 
              self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else None], ### 2
              ###[self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else self.double_out_filter_num[tokens[5]],
              ###self.double_out_filter_num[tokens[6]],
              ###self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else None], ### 3
              [self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else self.double_out_filter_num[tokens[4]], 
              self.double_out_filter_num[tokens[5]],
              self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else None, 2], ### 4
              [self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else self.double_out_filter_num[tokens[5]], 
              self.double_out_filter_num[tokens[6]],
              self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else None], ### 5
              ###[self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else self.double_out_filter_num[tokens[11]],
              ###self.double_out_filter_num[tokens[12]],
              ###self.double_filter_num[tokens[3]] if self.double_filter_num[tokens[3]] != 0 else None], ### 6
              ]
        self.with_extra_blocks = True if self.with_extra_blocks_num[tokens[7]] == 1 else False

        print('self.blaze_filters: ', self.blaze_filters)
        print('self.double_blaze_filters: ', self.double_blaze_filters)
                               
        def net_arch(input, mode, cfg):  #input is feed_vars if loss == ssd_loss
            self.output_decoder = cfg.BlazeFace['output_decoder']
            self.min_sizes = cfg.BlazeFace['min_sizes']
            self.use_density_prior_box = cfg.BlazeFace['use_density_prior_box']

            my_backbone = BlazeNet(blaze_filters=self.blaze_filters, double_blaze_filters=self.double_blaze_filters, 
                                   with_extra_blocks=self.with_extra_blocks, lite_edition=self.lite_edition)
            my_blazeface = BlazeFace(my_backbone, output_decoder=self.output_decoder, min_sizes=self.min_sizes, use_density_prior_box=self.use_density_prior_box) 
            return my_blazeface.build(input, mode=mode) 

        return net_arch
