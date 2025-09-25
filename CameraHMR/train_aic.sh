# CUDA_VISIBLE_DEVICES=3,4,5,6 python process_foundation_input_features.py data=train experiment=camerahmr exp_name=train_run1  trainer=ddp trainer.accelerator=gpu trainer.devices=4

CUDA_VISIBLE_DEVICES=3 python train.py data=aic experiment=camerahmr_multi exp_name=train_run3  trainer=ddp trainer.accelerator=gpu trainer.devices=2 +trainer.limit_val_batches=0.0 trainer.val_check_interval=null +trainer.check_val_every_n_epoch=None 2>&1 | tee train_run3.log
