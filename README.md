
# DETR Custom Pipeline

A custom version of DETR with working docker and layer freezing implementation. 

The main challenges
to DETR training are the very slow nature of transformer training and the increased memory usage
resulting if not controlled in the termination of the training process.

The original DETR implementation is openly available in the GitHub repository. For Original DETR README visit [this link](README_ORIGINAL.md)

## Usage

Use Dockerfile to build the environment. Prepare Dataset in COCO format and follow the instructions in the original DETR README. The training, with the COCO pretrained weights downloaded from the repository, can be started using the main.py script with the COCO styled dataset.

## Results

## Models

* Best Pretrained Model [D5](https://drive.google.com/file/d/1G84ybh_JvDLgcF-1OWM2ge63e3fCoESb/view?usp=sharing) 

* Similarly good Pretrained Model [D6](https://drive.google.com/file/d/17XW5SPGvE9HOQHyVpQjchYuD8PlTpXOF/view?usp=sharing)