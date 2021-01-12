import os
import sys
import cv2
from matplotlib import pyplot as plt
import sys
import json
import numpy as np
import base64
import tqdm
import args 
import face_alignment
from PIL import Image, ImageDraw
from argparse import ArgumentParser
from masked_face_sdk.mask_generation_utils import generate_masks_base
import face_alignment
import json
import args
from masked_face_sdk.mask_generation_utils import \
(
    extract_target_points_and_characteristic, 
    extract_polygon,
    rotate_image_and_points,
    draw_landmarks,
    warp_mask,
    get_traingulation_mesh_points_indexes,
    end2end_mask_generation
)



def gen_landmask(image_path, save_path):        
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    example_image = cv2.cvtColor(
        cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    landmarks = fa.get_landmarks_from_image(example_image)
    landmarks = np.floor(landmarks[0]).astype(np.int32)
    image = draw_landmarks(example_image, landmarks, color=(255, 0, 0), thickness=2)
    cv2.imwrite(save_path,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main(): 
    dataset_path ='/home/minglee/Documents/aiProjects/git_clone/wear_mask/dataset_path'
    save_dataset_path = '/home/minglee/Documents/aiProjects/git_clone/wear_mask/new_out'
    # dataset_path = './lfw'
    # save_dataset_path = './lfw_masked'
    unmasked_paths=[]
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for dir in tqdm.tqdm(dirs):
            fs = os.listdir(root + '/' + dir)
            for name in fs:
                new_root = root.replace(dataset_path, save_dataset_path)
                new_root = new_root + '/' + dir
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                # deal
    
                imgpath = os.path.join(root,dir, name)
                save_imgpath = os.path.join(new_root,name)
                gen_landmask(imgpath, save_imgpath)
    
if __name__ == '__main__':
    main()