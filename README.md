## Dataset download 
Images can be downloaded from here: https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing

Precomputed density maps can be found here: https://archive.org/details/FSC147-GT

Place the unzipped image directory and density map directory inside the data directory.

## Installation with Conda
#Attention Enhanced Multi-Scale Feature Map Fusion Few Shot Learning
conda create -n fscount python=3.7 -y

conda activate fscount

python -m pip install matplotlib opencv-python notebook tqdm

conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch


## Evaluation
We are providing our pretrained FamNet model, and the evaluation code can be used without the training.
### Testing on validation split without adaptation
```bash 
python test.py --data_path /PATH/TO/YOUR/FSC147/DATASET/ --test_split val
```
### Testing on val split with adaptation
```bash 
python test.py --data_path /PATH/TO/YOUR/FSC147/DATASET/ --test_split val --adapt
```


## Training 

python train.py 

