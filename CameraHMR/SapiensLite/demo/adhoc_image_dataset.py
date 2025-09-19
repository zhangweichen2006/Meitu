# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2

class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, shape=None, mean=None, std=None, cropping=False, no_padding=False, out_names=None, swapHW=False):
        self.image_list = image_list
        self.out_names = out_names
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None
        self.cropping = cropping
        self.no_padding = no_padding
        self.swapHW = swapHW
    def __len__(self):
        return len(self.image_list)

    def _preprocess(self, img):
        if self.swapHW:
            # rotate 90 degrees
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        if self.shape:
            target_height, target_width = self.shape
            if self.cropping:
                # Center-crop to target (height, width) instead of resizing
                image_height, image_width = img.shape[:2]

                # 竖屏图
                if image_height > target_height:
                    if image_height > image_width:
                        image_height_ratio = image_height / target_height
                        image_width_ratio = image_width / target_width
                        if image_height_ratio > 1.5 or image_width_ratio > 1.5:
                            # 竖屏图，且比例大于1.5，则resize较小ratio后进行中心裁剪
                            ratio = min(image_height_ratio, image_width_ratio)
                            new_height = int(image_height / ratio)
                            new_width = int(image_width / ratio)
                            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                            start_y = (new_height - target_height) // 2
                            start_x = (new_width - target_width) // 2
                            end_y = start_y + target_height
                            end_x = start_x + target_width
                            img = img[start_y:end_y, start_x:end_x]
                        else:
                            # 竖屏图，且比例小于1.5，则直接crop中心区域
                            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                else:
                    # 横屏图
                    image_height_ratio = image_height / target_height
                    image_width_ratio = image_width / target_width

                crop_height = min(target_height, image_height)
                crop_width = min(target_width, image_width)

                if self.no_padding:
                    if target_height > image_height or target_width > image_width:
                        # fix ratio crop
                        ratio = max(target_height / image_height, target_width / image_width)
                        new_height = int(target_height / ratio)
                        new_width = int(target_width / ratio)

                        # crop center new_height, new_width of the image
                        start_y = (image_height - new_height) // 2
                        start_x = (image_width - new_width) // 2
                        end_y = start_y + new_height
                        end_x = start_x + new_width
                        img = img[start_y:end_y, start_x:end_x]

                        # resize to target_height, target_width
                        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                    else:
                        # just crop the middle target_height, target_width of the image
                        start_y = (image_height - target_height) // 2
                        start_x = (image_width - target_width) // 2
                        end_y = start_y + target_height
                        end_x = start_x + target_width
                        img = img[start_y:end_y, start_x:end_x]

                else:
                    # if target_height > image_height, do padding of top and bottom, centered on image center
                    start_y = (image_height - crop_height) // 2
                    start_x = (image_width - crop_width) // 2
                    end_y = start_y + crop_height
                    end_x = start_x + crop_width
                    img = img[start_y:end_y, start_x:end_x]

                    if not self.no_padding and target_height > image_height:
                        padding_height = (target_height - image_height) // 2
                        img = cv2.copyMakeBorder(img, padding_height, padding_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    if target_width > image_width:
                        padding_width = (target_width - image_width) // 2
                        img = cv2.copyMakeBorder(img, 0, 0, padding_width, padding_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean=self.mean.view(-1, 1, 1)
            std=self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img

    def __getitem__(self, idx):
        orig_img_dir = self.image_list[idx]
        out_img_dir = self.out_names[idx]
        orig_img = cv2.imread(orig_img_dir)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = self._preprocess(orig_img)
        return orig_img_dir, out_img_dir, orig_img, img
