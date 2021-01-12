from argparse import ArgumentParser
from masked_face_sdk.mask_generation_utils import generate_masks_base
import face_alignment
import json
import args

# def parse_args():
#     parser = ArgumentParser(description='Generate masks database')
#
#     parser.add_argument(
#         '--masks-folder', required=True, type=str, default='/home/minglee/Documents/aiProjects/git_clone/face-id-with-medical-masks/folderImagesWithMask',
#         help='Path to folder with masks images.'
#     )
#
#     parser.add_argument(
#         '--database-file', required=True, type=str,default='/home/minglee/Documents/aiProjects/git_clone/face-id-with-medical-masks/folderCreateJsonFIle',
#         help='Path to created json database file.'
#     )
#
#     parser.add_argument(
#         '--verbose', action='store_true'
#     )
#
#     parser.add_argument(
#         '--skip-warnings', action='store_true'
#     )
#
#     return parser.parse_args()


def main():

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device='cuda'
    )

    data_dict = generate_masks_base(
        args.masks_folder,
        fa,
        args.verbose,
        args.skip_warnings
    )

    with open(args.database_file, 'w') as jf:
        json.dump(data_dict, jf, indent=4)

    print(
        'Masks database successfully saved by follow path: {}'.format(
            args.database_file
        )
    )


if __name__ == '__main__':
    main()
