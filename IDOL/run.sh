# /bin/bash

source /home/cevin/miniconda3/etc/profile.d/conda.sh
conda activate hper
cd /home/cevin/Meitu/CameraHMR/
python demo.py --image_folder /home/cevin/Meitu/IDOL/test_data_img/all --output_folder /home/cevin/Meitu/IDOL/test_data_img/out --output_cam

# conda activate idol
# python /home/cevin/Meitu/IDOL/run_demo.py --render_mode reconstruct