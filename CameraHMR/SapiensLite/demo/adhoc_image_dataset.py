# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2

class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, shape=None, mean=None, std=None, cropping=False, resize=False, out_names=None, out_imgmatch_names=None, swapHW=False, zoom_to_3Dpt=False):
        self.image_list = image_list
        self.out_names = out_names
        self.out_imgmatch_names = out_imgmatch_names
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None
        self.cropping = cropping
        self.resize = resize
        self.zoom_to_3Dpt = zoom_to_3Dpt
        self.swapHW = swapHW
    def __len__(self):
        return len(self.image_list)

    def _preprocess(self, img):
        if self.swapHW:
            # rotate 90 degrees
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        if self.shape:
            image_height, image_width = img.shape[:2]
            target_height, target_width = self.shape

            if self.cropping:
                # Center crop to (target_height, target_width). If the crop goes
                # outside image bounds, pad with zeros to keep the window centered.
                if image_height > image_width:
                    # Portrait verticle image, crop and resize

                    # Scale First (but minimal change) then Crop
                    if target_height / image_height > (4./3.) or target_width / image_width > (4./3.):
                        # resize first based on smaller ratio then do following crop
                        resize_ratio = min(target_height / image_height, target_width / image_width)
                        img = cv2.resize(img, (int(image_width * resize_ratio), int(image_height * resize_ratio)), interpolation=cv2.INTER_LINEAR)
                    elif target_height / image_height < 0.75 or target_width / image_width < 0.75:
                        # resize first based on larger ratio then do following crop
                        resize_ratio = max(target_height / image_height, target_width / image_width)
                        img = cv2.resize(img, (int(image_width * resize_ratio), int(image_height * resize_ratio)), interpolation=cv2.INTER_LINEAR)
                    else:
                        if self.resize:
                            # resize smaller ratio
                            resize_ratio = min(target_height / image_height, target_width / image_width)
                            img = cv2.resize(img, (int(image_width * resize_ratio), int(image_height * resize_ratio)), interpolation=cv2.INTER_LINEAR)

                    image_height, image_width = img.shape[:2]
                    start_y = (image_height - target_height) // 2
                    start_x = (image_width - target_width) // 2
                    end_y = start_y + target_height
                    end_x = start_x + target_width

                    pad_top = max(0, -start_y)
                    pad_left = max(0, -start_x)
                    pad_bottom = max(0, end_y - image_height)
                    pad_right = max(0, end_x - image_width)

                    if pad_top or pad_bottom or pad_left or pad_right:
                        img = cv2.copyMakeBorder(
                            img,
                            pad_top,
                            pad_bottom,
                            pad_left,
                            pad_right,
                            cv2.BORDER_CONSTANT,
                            value=[0, 0, 0],
                        )
                        # Adjust coordinates after padding
                        start_y += pad_top
                        start_x += pad_left
                        end_y = start_y + target_height
                        end_x = start_x + target_width

                    img = img[start_y:end_y, start_x:end_x]
                    if self.zoom_to_3Dpt:
                        # boost two times
                        img = cv2.resize(img, (target_width * 2, target_height * 2), interpolation=cv2.INTER_LINEAR)
                        # just pick center part
                        img = img[target_height//2:target_height//2+target_height, target_width//2:target_width//2+target_width]
                        # and scale back to target_width and target_height
                        # img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                else:
                    # Landscape horizontal image

                    # match width to target_width for optimum coverage
                    if self.resize or target_width / image_width > (4./3.) or target_width / image_width < 0.75 and not self.zoom_to_3Dpt:
                        resize_ratio = target_width / image_width
                        new_height = int(image_height * resize_ratio)
                        img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_LINEAR)

                    # After optional resize, center-crop if larger, pad if smaller
                    cur_h, cur_w = img.shape[:2]

                    # Vertical dimension: pad to target_height

                    if cur_h >= target_height:
                        start_y = (cur_h - target_height) // 2
                        end_y = start_y + target_height
                        img = img[start_y:end_y, :]
                        cur_h = target_height
                    else:
                        pad_total = target_height - cur_h
                        pad_top = pad_total // 2
                        pad_bottom = pad_total - pad_top
                        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        cur_h = target_height

                    if cur_w >= target_width:
                        start_x = (cur_w - target_width) // 2
                        end_x = start_x + target_width
                        img = img[:, start_x:end_x]
                        cur_w = target_width
                    else:
                        pad_total = target_width - cur_w
                        pad_left = pad_total // 2
                        pad_right = pad_total - pad_left
                        img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        cur_w = target_width

                    if self.zoom_to_3Dpt:
                        # boost two times
                        img = cv2.resize(img, (target_width * 2, target_height * 2), interpolation=cv2.INTER_LINEAR)
                        # just pick center part
                        img = img[target_height//2:target_height//2+target_height, target_width//2:target_width//2+target_width]
                        # and scale back to target_width and target_height
                        # img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
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
        orig_img = cv2.imread(orig_img_dir)
        img = self._preprocess(orig_img)

        out_img_dir = self.out_names[idx] if self.out_names else []
        out_imgmatch_dir = self.out_imgmatch_names[idx] if self.out_imgmatch_names else []
        return orig_img_dir, out_img_dir, out_imgmatch_dir, orig_img, img
