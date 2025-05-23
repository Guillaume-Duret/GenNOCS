#!/bin/bash
cd /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/
source /root/miniconda3/etc/profile.d/conda.sh
conda activate deep_sdf
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
Xvfb :$1 -screen 0 1024x768x24 &
sleep 2  # Ensure Xvfb is up and running before proceeding
export DISPLAY=:$1
export MESA_GL_VERSION_OVERRIDE=3.30
echo $3/data_deepsdf/
echo $3/train/
echo $2
echo $3/split/$2.json
echo "test0"
echo $1
ls "$3"
ls "$3/split"
echo "test"
#python /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/preprocess_data.py --data_dir $3/data_deepsdf/ \
#    --source $3/$2/ \
#    --name $2 \
#    --split $3/split/$2.json \
#    --skip \
#    --threads 360
python /home/input_data/deep_sdf_code_test/DeepSDF/preprocess_data.py --data_dir $3/data_deepsdf/ \
    --source $3/$2/ \
    --name $2 \
    --split $3/split/$4_$2.json \
    --surface \
    --skip \
    --threads 80

python /home/input_data/deep_sdf_code_test/DeepSDF/preprocess_data.py --data_dir $3/data_deepsdf/ \
    --source $3/$2/ \
    --name $2 \
    --split $3/split/$4_$2.json \
    --skip \
    --threads 80


