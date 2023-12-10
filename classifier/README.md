# classifier

## Purpose:

Fine tune a pre-trained deep learning classifier for new dataset

## General Tips

### Enroot

Make yourself a deploy token: 
- Found by navigating: settings -> repository -> deploy tokens
- Write down password

Import enroot sqsh file 
- `enroot import --output enroot_envs/classifier.sqsh docker://general@gitlab.cgi.missouri.edu:4567#nvesd/classifier/gen:latest`
- Enter the deploy token password (won't be able to see typed characters b/c terminal) but copy and paste also works well
- This operation only has to be done once and you can always ask some user of this repo for a copy of their sqsh file

Create the enroot image from sqsh file
- `enroot create -n classifier enroot_envs/classifier.sqsh`
- This operation has to be done once per machine using it, so recommending two times. Horizon VM is one, SLURM launch node is the other.

Run enroot image on Horizon VM 
- `enroot start -m /cgi/data/nvesd/organized/formatted/real/chips/20231023_2ndStage_Testing_Data/data:/develop/data -m code/classifier:/develop/code -m results:/develop/results -m $HOME:$HOME classifier`
- Since this is running on the VM, this is for analyzing the automation results:
    - Training / Valdation: Can monitor progress via python plotly dashboard 
    - Testing: Organizes results into json files for scoring outside of this library

## Training on Dataset


## Monitor Training


## Testing / Scoring


### SLURM - Make predictions for each trained model


### Horizon - Produce JSON for each trained model

Start a tmux session `tmux new -s scoring`, split accordingly `ctrl b + %`, and setup workstation
- Like above, I recommend code on the left and launching on the right
    - Open code: `vim code/classifier/classifier/utils/evaluation/log_confidence.py`
    - Enter enroot: `enroot start -m code/classifier:/develop/code -m results:/develop/data -m /cgi/data/nvesd/organized/formatted/real/chips/20231206_2ndStage_TestingData/data:/develop/results classifier`

Run logger to produce JSON files 
- `python classifier/utils/evaluation/log_confidence.py`


