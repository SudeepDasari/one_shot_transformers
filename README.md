# Transformers for One-Shot Imitation Learning
## Installation
This code was developed using python 3.7 on a CUDA 10 machine. For convenience, a singularity definition file - `singularity.def` - is provided which can replicate the GPU environment required for headless MuJoCo rendering alongside our pytorch codebase. You can find directions for how to create and use singularity "sandboxes" [here](https://singularity.lbl.gov/docs-build-container#creating---writable-images-and---sandbox-directories).

Once you've setup a suitable environment, please install [Robosuite](https://github.com/SudeepDasari/robosuite). Robosuite has changed quite a bit compared to the version used by this project, so to be sure to use the fork I linked rather than downloading it yourself. Now run:
```
cd path/to/one_shot_transformers
pip install -r requirements.txt
python setup.py develop
```

## Usage Instructions
These basic usage instructions will allow you to generate our base dataset, train our model, and replicate the main result in our experiments section. 
```
# collect training dataset
mkdir dataset && cd dataset
python path/to/one_shot_transformers/scripts/collect_demonstrations.py --env PandaPickPlaceDistractor --num_workers 30 --N 1600 --collect_cam --per_task_group 100 --n_env 800 ./panda
python path/to/one_shot_transformers/scripts/collect_demonstrations.py --env SawyerPickPlaceDistractor --num_workers 30 --N 1600 --collect_cam --per_task_group 100 --n_env 800 ./sawyer

# train model
mkdir model && cd model
export EXPERT_DATA=/path/to/dataset
export CUDA_VISIBLE_DEVICES=0,1
python path/to/one_shot_transformers/scripts/train_transformer.py path/to/one_shot_transformers/experiments/base.yaml

# test model
python path/to/one_shot_transformers/scripts/test_transformers.py path/to/model_save-60000.pt --N 100 --num_workers 10
```

## Citation
If this code was useful please conider citing our paper:
```
@inproceedings{dasari2020transformers,
    title={Transformers for One-Shot Imitation Learning},
    author={Sudeep Dasari and Abhinav Gupta},
    year={2020},
    primaryClass={cs.RO},
    booktitle={CoRL 2020}
}

```
