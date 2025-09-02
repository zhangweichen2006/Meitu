import os, sys
# os.environ["WANDB_MODE"] = "dryrun" # default setting to save locally
from lib.utils.train_util import main_print
import torch
# Check GPU information
if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_name()
    if "H20" in gpu_info or "H800" in gpu_info:
        os.environ["NCCL_SOCKET_IFNAME"] = "bond1" # for H20  # If using H20 GPU, set network interface
        main_print("changing the network interface to bond1")
    if "H800" in gpu_info:
        # Set precision for matrix multiplication
        torch.set_float32_matmul_precision('medium')  # or 'high'
    

import argparse
import shutil
import subprocess
from omegaconf import OmegaConf

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from lib.utils.train_util import instantiate_from_config

from pytorch_lightning import loggers as pl_loggers


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="only resume model weights",
    )
    parser.add_argument(
        "--resume_not_loading_decoder",
        action="store_true",
        help="only resume model weights excepts decoder",
    )
    # parser.add_argument(
    #     "--custom_loading_for_PA",
    #     action="store_true",
    #     help="customly loading the PA network",
    # )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="experiment name",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,",
        help="gpu ids to use",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging data",
    )
    parser.add_argument(
        "--test_sd",
        type=str,
        default="",
        help="path to state dict for testing",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="./configs/test_dataset.yaml",
        help="path to state dict for testing",
    )
    parser.add_argument(
        "--is_debug",
        action="store_true",
        help="flag to specify if in debug mode, if true, it will returns more results",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default=None,
        help="flag to specify the training strategy",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            main_print("Project config")
            main_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))


class CodeSnapshot(Callback):
    """
    Modified from https://github.com/threestudio-project/threestudio/blob/main/threestudio/utils/callbacks.py#L60
    """
    def __init__(self, savedir, exclude_patterns=None):
        self.savedir = savedir
        # Default excluded files and folders patterns
        self.exclude_patterns = exclude_patterns or [
            "*.mp4", "*.npy", "work_dirs/*", "processed_data/*", "logs/*"
        ]

    def get_file_list(self):
        # Get git tracked files, excluding configs directory
        tracked_files = subprocess.check_output(
            'git ls-files -- ":!:configs/*"', shell=True
        ).splitlines()
        
        # Get untracked but not ignored files
        untracked_files = subprocess.check_output(
            "git ls-files --others --exclude-standard", shell=True
        ).splitlines()
        
        # Merge file lists and decode
        all_files = [b.decode() for b in set(tracked_files) | set(untracked_files)]
        
        # Apply exclusion pattern filtering
        filtered_files = []
        for file_path in all_files:
            should_exclude = False
            for pattern in self.exclude_patterns:
                if self._match_pattern(file_path, pattern):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_files.append(file_path)
                
        return filtered_files
    
    def _match_pattern(self, file_path, pattern):
        """Check if file path matches the given pattern"""
        # Handle directory wildcard patterns (e.g., work_dirs/*)
        if pattern.endswith('/*'):
            dir_prefix = pattern[:-1]  # Remove '*'
            return file_path.startswith(dir_prefix)
        
        # Handle file extension patterns (e.g., *.mp4)
        if pattern.startswith('*'):
            ext = pattern[1:]  # Get extension part
            return file_path.endswith(ext)
        
        # Exact matching
        return file_path == pattern

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            main_print(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


if __name__ == "__main__":
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, cfg_name+exp_name)


    # init configs
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    trainer_config = lightning_config.trainer

    # modify some config for debug mode
    if opt.is_debug:

        lightning_config['trainer']['val_check_interval'] = 1
        exp_name = 'debug'
        logdir = os.path.join(opt.logdir, cfg_name+exp_name)
        config.model.params['is_debug'] = True
        config.dataset.batch_size = 1 #ss
        config.dataset.num_workers = 1
        config.dataset.params.train.params.cache_path = config.dataset.params.debug_cache_path

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")
    seed_everything(opt.seed)
    

    main_print(f"Running on GPUs {opt.gpus}")
    ngpu = len(opt.gpus.strip(",").split(','))
    trainer_config['devices'] = ngpu

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # testing setting
    if len(opt.test_sd) > 0:
        config_dataset = OmegaConf.load(opt.test_dataset)
        config.dataset = config_dataset.dataset


    precision_config = {'precision':"bf16"}

    # model
    model = instantiate_from_config(config.model)
    if precision_config['precision'] == "bf16":
        model.encoder = model.encoder.to(torch.bfloat16)
    if opt.resume and opt.resume_weights_only:
        if opt.resume_not_loading_decoder:
            main_print("========Loading only model weights excepts decoder ==============")
            # Load complete state dictionary
            state_dict = torch.load(opt.resume, map_location='cpu')['state_dict']
            # Create a new state dictionary only containing the parts you want to load
            new_state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('encoder') or k.startswith('decoder') or k.startswith('lpips'))}
            # Load the remaining state dictionary
            model.load_state_dict(state_dict, strict=False)
            del state_dict


   
        with torch.amp.autocast( device_type='cpu'):
            state_dict = torch.load(opt.resume, map_location='cpu')['state_dict']
            main_print([k for k in state_dict.keys()  if not k.startswith('lpips') ])
            new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('lpips')}
            model.load_state_dict(new_state_dict, strict=False)
        model = model.to('cuda')


    model.logdir = logdir

    # trainer and callbacks
    trainer_kwargs = dict()

    # logger
    param_log = { 
        'save_dir': logdir,
        'name': cfg_name+exp_name,
    }
    trainer_kwargs["logger"] = [ 
        pl_loggers.TensorBoardLogger(**param_log),
      pl_loggers.CSVLogger(**param_log)

    ]

    # model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 5000,
            "save_top_k": -1,   # save all checkpoints
        }
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "code_snapshot": {
            "target": "train.CodeSnapshot",
            "params": {
                "savedir": codedir,
            }
        },
    }
    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]



    training_mode = "DDP" if  opt.training_mode is None else  opt.training_mode

    if training_mode == 'DDP':
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False, static_graph=True) # TODO modify to True
    elif training_mode == 'ZERO':
        #  DeepSpeed 
        strategy = DeepSpeedStrategy(config='./configs/deepspeed_config.json')
        trainer_kwargs["strategy"] = strategy# TODO modify to True
    elif training_mode == 'FSDP':
        from pytorch_lightning.strategies import FSDPStrategy
        fsdp_strategy = FSDPStrategy(
            auto_wrap_policy=None,  
            activation_checkpointing_policy=None,  
            cpu_offload=False,  # Whether to offload model parameters to CPU
            limit_all_gathers=False,  # Whether to limit all gather operations
            sync_module_states=True,  # Whether to synchronize module states
            # use_sharded_checkpoint=True,  # Whether to use sharded checkpoints
            mixed_precision='bf16',  # Mixed precision training, default is 'bf16'
        )
        trainer_kwargs["strategy"] = fsdp_strategy
    main_print(f" ............ trying training strategy {training_mode} ...........")



    trainer = Trainer(**precision_config, **trainer_config, **trainer_kwargs, num_nodes=opt.num_nodes)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.dataset)
    data.prepare_data()
    data.setup("fit")
    
    
    # configure learning rate
    base_lr = config.model.params.neck_learning_rate
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    main_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    main_print("++++ NOT USING LR SCALING ++++")
    main_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # trainer.fit(model, data) # debug 
    
    if len(opt.test_sd) > 0:
        sd = torch.load(opt.test_sd, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        with torch.amp.autocast(device_type='cpu'):
            # import ipdb; ipdb.set_trace()
            def load_folder_ckpt(checkpoint_dir):
                # For DeepSpeed loading
                # Get all .pt files
                pt_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                # Initialize model state dictionary
                model_state_dict = {}
                # Load each .pt file and merge into model state dictionary
                for pt_file in pt_files:
                    state_dict = torch.load(pt_file, map_location='cpu')
                    model_state_dict.update(state_dict)
                return model_state_dict
            if os.path.isdir(opt.test_sd):
                state_dict = load_folder_ckpt(opt.test_sd+"/checkpoint")
                # Load checkpoint
                success = model.load_checkpoint(opt.test_sd, load_optimizer_states=True, load_lr_scheduler_states=True)

            else:
                state_dict = torch.load(opt.test_sd, map_location='cpu')['state_dict']
            main_print([k for k in state_dict.keys()  if not k.startswith('lpips') ])
            new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('lpips')}
            new_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith('encoder')}
            model.load_state_dict(new_state_dict, strict=False)
            main_print(f"========testing =====, loading from {opt.test_sd} ================")
        model = model.to('cuda')

        with torch.no_grad():
            trainer.test(model, data)
            
    else:
        # run training loop
        try:
            if opt.resume and not opt.resume_weights_only:
                trainer.fit(model, data, ckpt_path=opt.resume)
            else:
                trainer.fit(model, data)
        except  Exception as e:
            main_print(f"An error occurred: {e}")
            torch.cuda.empty_cache()
