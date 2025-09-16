CUDA_VISIBLE_DEVICES=4,5 python train.py data=train experiment=camerahmr exp_name=train_run1  trainer=ddp trainer.accelerator=gpu trainer.devices=2
