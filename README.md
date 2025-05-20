
# Usage
## 1. Create Environment
* create conda environment
```
conda create -n LDFusion python=3.10
conda activate LDFusion
```

* Install Dependencies 
```
pip install -r requirements.txt
```
(recommended cuda11.6 and torch 1.13.1)

## 2. Data Preparation and Running
Please put test data into the ```test_imgs``` directory (infrared images in ```ir``` subfolder, visible images in ```vi``` subfolder), and run ```python src/test.py```. 

(Note: The weight files (*.pt) might require independent download from the repository)

Then, the fused results will be saved in the ```./results/``` folder. 


