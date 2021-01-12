import os
import sys
import cv2
from matplotlib import pyplot as plt
import sys
import json
import tqdm
import numpy as np
import base64
import args 
import face_alignment
from PIL import Image, ImageDraw

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
def main():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    idx_image = 2  # chọn image cần dự doán.  idx phải nhỏ hơn tổng số image hiện có. 
    folder_path = '/home/minglee/Documents/aiProjects/git_clone/face-id-with-medical-masks/imageNoneMask'
    masked_faces_images_pathes = [
        os.path.join(folder_path, img_name)
        for img_name in os.listdir(folder_path)
    ]
    print("List folder of image ", masked_faces_images_pathes)
    example_image = cv2.cvtColor(
        cv2.imread(
            masked_faces_images_pathes[idx_image]
        ), 
        cv2.COLOR_BGR2RGB
    )
    print("take image for example ",masked_faces_images_pathes[idx_image])
    landmarks = fa.get_landmarks_from_image(example_image)
    landmarks = np.floor(landmarks[0]).astype(np.int32)
    
    """
    print landmarks of face, its points of face alignment
    """
    
    plt.figure(figsize=(8, 8))
    plt.imshow(example_image)
    plt.axis('off')
    
    plt.figure(figsize=(8, 8))
    plt.imshow(draw_landmarks(example_image, landmarks, color=(255, 0, 0), thickness=2))
    plt.axis('off')
    plt.show()
    
    """
    this scrip for generate face wear mask 
    """
    
    target_points, s1, s2 = extract_target_points_and_characteristic(landmarks)
    mask_rgba_crop, target_points = extract_polygon(example_image, target_points)
    
    
    mask_rgba_crop, target_points = rotate_image_and_points(mask_rgba_crop, s1, target_points)
    
    triangles_indexes = get_traingulation_mesh_points_indexes(target_points)
    
    mask_rgba_crop_vis = mask_rgba_crop[..., :3].copy().astype(np.uint8)
    # for triangle in triangles_indexes:
    #     triangle_points = target_points[triangle]
    # 
    #     mask_rgba_crop_vis = cv2.line(
    #         mask_rgba_crop_vis,
    #         tuple(triangle_points[0]),
    #         tuple(triangle_points[1]),
    #         (0, 255, 0),
    #         1
    #     )
    # 
    #     mask_rgba_crop_vis = cv2.line(
    #         mask_rgba_crop_vis,
    #         tuple(triangle_points[1]),
    #         tuple(triangle_points[2]),
    #         (0, 255, 0),
    #         1
    #     )
    # 
    #     mask_rgba_crop_vis = cv2.line(
    #         mask_rgba_crop_vis,
    #         tuple(triangle_points[0]),
    #         tuple(triangle_points[2]),
    #         (0, 255, 0),
    #         1
    #     )
    
    """
    print image crop mask and landmarks crop. 
    """
    # plt.figure(figsize=(8, 8))
    # plt.imshow(draw_landmarks(mask_rgba_crop_vis, target_points, color=(255, 0, 0), thickness=4))
    # plt.axis('off')
    # 
    # plt.figure(figsize=(8, 8))
    # plt.imshow(mask_rgba_crop[..., 3], 'gray')
    # plt.axis('off')
    # 
    # 
    # plt.show()
    # 
    # #@title <b><font color="red" size="+3">←</font><font color="black" size="+3"> Face matching example</font></b>
    path_example = '/home/minglee/Documents/aiProjects/git_clone/face-id-with-medical-masks/imageNoneMask/7.jpg'
    target_image = cv2.cvtColor(cv2.imread(
        path_example, 
        cv2.IMREAD_COLOR
        ),
        cv2.COLOR_BGR2RGB
    )
    
    landmarks2 = fa.get_landmarks_from_image(target_image)
    landmarks2 = np.floor(landmarks2[0]).astype(np.int32)
    
    target_points2, _, _ = extract_target_points_and_characteristic(landmarks2)
    target_image_with_mask = warp_mask(
        mask_rgba_crop[..., :3],
        target_image,
        target_points,
        target_points2
    )
    
    
    triangles_indexes = get_traingulation_mesh_points_indexes(target_points2)
    
    mask_rgba_crop_vis = target_image_with_mask.copy().astype(np.uint8)
    for triangle in triangles_indexes:
        triangle_points = target_points2[triangle]
    
        mask_rgba_crop_vis = cv2.line(
            mask_rgba_crop_vis,
            tuple(triangle_points[0]),
            tuple(triangle_points[1]),
            (0, 255, 0),
            3
        )
    
        mask_rgba_crop_vis = cv2.line(
            mask_rgba_crop_vis,
            tuple(triangle_points[1]),
            tuple(triangle_points[2]),
            (0, 255, 0),
            3
        )
    
        mask_rgba_crop_vis = cv2.line(
            mask_rgba_crop_vis,
            tuple(triangle_points[0]),
            tuple(triangle_points[2]),
            (0, 255, 0),
            3
        )
    
    """"
    this print image root with figure 1 
    print image landmarks with figure 2 
    print image target image with mask figure 3
    """
    # plt.figure(figsize=(12, 12))
    # plt.imshow(target_image)
    # plt.axis('off')
    # 
    # plt.figure(figsize=(12, 12))
    # plt.imshow(draw_landmarks(mask_rgba_crop_vis, target_points2, color=(255, 0, 0), thickness=4))
    # plt.axis('off')
    # 
    # plt.figure(figsize=(12, 12))
    # plt.imshow(target_image_with_mask)
    # plt.axis('off')
    # plt.show()
    # 
    
    """
    scrip below generate mask by image with json of points. 
    """
    
    # try to generator image by mask 
    # path_example = '/home/minglee/Documents/aiProjects/git_clone/face-id-with-medical-masks/imageNoneMask/3.jpg'
    # target_image = cv2.cvtColor(cv2.imread(
    #     path_example, 
    #     cv2.IMREAD_COLOR
    #     ),
    #     cv2.COLOR_BGR2RGB
    # )
    # 
    # path_json_file = '/home/minglee/Documents/aiProjects/git_clone/face-id-with-medical-masks/folderCreateJsonFIle/data.json'
    # with open(path_json_file, 'r') as jf:
    #     masks_database = json.load(jf)
    # target_image_with_mask = end2end_mask_generation(target_image, masks_database, fa)
    # """
    # print image wear mask . 
    # """
    # 
    # plt.figure(figsize=(12, 12))
    # plt.imshow(target_image_with_mask)
    # plt.axis('off')
    # plt.show()
    

if __name__ == '__main__':
    main()