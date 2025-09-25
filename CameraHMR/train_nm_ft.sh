# CUDA_VISIBLE_DEVICES=3,4,5,6 python process_foundation_input_features.py data=train experiment=camerahmr exp_name=train_run1  trainer=ddp trainer.accelerator=gpu trainer.devices=4

CUDA_VISIBLE_DEVICES=3,4,5,6 python train.py data=process experiment=camerahmr_multi_crossAttn exp_name=train_run_nm_lora_ft  trainer=ddp trainer.accelerator=gpu trainer.devices=2 ++DATASETS.check_file_completeness_and_filter=True ++trainer.val_check_interval=null ++trainer.limit_val_batches=0.0 ++DATASETS.check_file_completeness_and_filter=True 2>&1 | tee train_run_nm_lora_ft.log
