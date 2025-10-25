from __future__ import division, print_function, absolute_import
import os
import copy
from pkd.data_loader.incremental_datasets import IncrementalFaceSamples
import re
import glob
import os.path as osp
import warnings
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random

class IncrementalSamples4umdface(IncrementalFaceSamples):
    def __init__(self, dataset_root, relabel=True):
        self.dataset_root = os.path.join(dataset_root, 'UMDFace')
        print("dataset path:",self.dataset_root)
        self.relabel = relabel  # Add relabel parameter
        self.train, self.query, self.gallery = self.get_image_paths_and_labels()
        self._show_info(self.train, self.query, self.gallery)

    def get_image_paths_and_labels(self):
        tr_data = []
        query_data = []
        gallery_data = []

        if self.relabel:
            label_mapping = {}  # Create a mapping from original pid to relabeled pid

        for i, person_dir in enumerate(os.listdir(self.dataset_root)):
            person_path = os.path.join(self.dataset_root, person_dir)
            if os.path.isdir(person_path):
                person_all_image_path = [os.path.join(person_path, image_file) for image_file in os.listdir(person_path) if image_file.endswith('.jpg')]
                for image_file in os.listdir(person_path):
                    if image_file.endswith('.jpg'):
                        if self.relabel:
                            if int(person_dir) not in label_mapping:
                                label_mapping[int(person_dir)] = len(label_mapping)
                            pid = label_mapping[int(person_dir)]
                        else:
                            pid = int(person_dir)
                train_images, test_images = train_test_split(person_all_image_path, test_size=0.3, random_state=42)
                for tr_img_path in train_images:
                    tr_data.append([tr_img_path, pid, 'umdface'])
                query_img_idx = random.randint(0, len(test_images)-1)
                for j, img_path in enumerate(test_images):
                    if j == query_img_idx: 
                        query_data.append([img_path, pid, 'umdface'])
                    else:
                        gallery_data.append([img_path, pid, 'umdface'])

        return tr_data, query_data, gallery_data
