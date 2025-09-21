CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop-local/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221017_3_1000_batch01hand_6fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop-local/20221017_3_1000_batch01hand_6fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop-local/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop-local/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=7 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221019_3_250_highbmihand_6fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop-local/20221019_3_250_highbmihand_6fps \
    --batch_size 1