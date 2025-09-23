# CUDA_VISIBLE_DEVICES=3,4,5,6 python process_foundation_input_features.py data=train experiment=camerahmr exp_name=train_run1  trainer=ddp trainer.accelerator=gpu trainer.devices=4

CUDA_VISIBLE_DEVICES=3,4,5,6 python train.py data=train_mod experiment=camerahmr_multi exp_name=train_run2  trainer=ddp trainer.accelerator=gpu trainer.devices=4
