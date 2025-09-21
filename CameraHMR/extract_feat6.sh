CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221019_3-8_1000_highbmihand_static_suburb_d_6fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop/20221019_3-8_1000_highbmihand_static_suburb_d_6fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221022_3_250_batch01handhair_static_bigOffice_30fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop/20221022_3_250_batch01handhair_static_bigOffice_30fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps \
    --batch_size 1

CUDA_VISIBLE_DEVICES=6 python SapiensLite/demo/vis_normal.py \
    SapiensLite/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2 \
    --input data/training-images/20221024_10_100_batch01handhair_zoom_suburb_d_30fps \
    --output_root data/training-images-sapiens-normals-OrgPadCrop/20221024_10_100_batch01handhair_zoom_suburb_d_30fps \
    --batch_size 1
# 20221019_1_250_highbmihand_closeup_suburb_c_6fps