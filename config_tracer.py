import os

detector_path = r'/home/upload_code/detectorhit'
tracer_path = r'/home/upload_code/tracer'

# detector_path = r'/home/cuiyang/pcl/docker_code/detectorhit'
# tracer_path = r'/home/cuiyang/pcl/docker_code/tracer'

# detector_path = r'/code/detectorhit'
# tracer_path = r'/code/tracer'

config_tracer={
    'detector_path': detector_path,
    'feat_save_path': os.path.join(detector_path, 'new_result_for_feat'), # not used
    'tracer_path': tracer_path,

    'cover_model': os.path.join(tracer_path, 'ckpt_cover'),
    'class_model': os.path.join(tracer_path, 'ckpt_class'),
    'optimizer_model': os.path.join(tracer_path, 'optimizer_ckpt'),

    'input_path': os.path.join(detector_path, 'new_result_for_feat'), # not used
    'output_path': os.path.join(tracer_path, 'cover_2d_out'),

    'zipfiles': os.path.join(tracer_path, 'upload_files'),
    'extractfiles': os.path.join(tracer_path, 'extractfiles'),
    'flush_zip': True,
}
