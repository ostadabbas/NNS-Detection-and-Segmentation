# from datasets.kinetics import Kinetics
# from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101


# from datasets.hmdb51 import HMDB51


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    if opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    if opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data


# def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
#     if opt.dataset == 'ucf101':
#         test_data = UCF101(
#             opt.video_path,
#             opt.annotation_path,
#             'testing',
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)
#
#     return test_data
