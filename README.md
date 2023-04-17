# ProtRAP
Deep learning based protein relative accessibility predictor to predict the relative lipid accessibility and relative solvent accessibility of residues from a given protein sequence
## Introduction
Solvent accessibility has been extensively used to characterize and predict the chemical properties of the surface residues of soluble proteins. However, there is not yet a widely accepted quantity of the same dimension for the study of lipid-accessible residues of membrane proteins. Here we propose that lipid accessibility, defined in a similar way to solvent accessibility, can be used to characterize the lipid accessible residues of membrane proteins. [Details can be found here](https://chemrxiv.org/engage/chemrxiv/article-details/6305cbd0f07ee1b928efade2)

![TopOfContent](https://github.com/ComputBiophys/ProtRAP/releases/download/weights/TopOfContent.png)

In models.py, we provide the definition and implementation of the final model (Transformer light).

driver.py is a simple demonstration of how to process input data and process models

prot.feat, prot.fasta are example files
### Quick start
We provide a prediction server to meet researchers' small batch sample prediction needs.
  
  http://www.songlab.cn/ProtRAP/Introduction/
### Requirements
* PyTorch
* NumPy
### Feature generation
Our model requires One-hot encoded sequence information (20 bits), PSSM (20 bits) and predicted SS3 (3 bits) as input features.

The order of One-hot encoding is: ACDEFGHIKLMNPQRSTVWY,
we provide the seq2arr function in the driver.py file

PSSM and predicted SS3 were generated by [RaptorX-Property](https://github.com/realbigws/RaptorX_Property_Fast) aligning against database [uniclust30_2017_10](http://wwwuser.gwdg.de/~compbiol/uniclust/2017_10/). The file suffix is .feat, and we provide the load_feat function in the driver.py file
### Usage
First download the weights file we provided in Releases. Then use `torch.load(absolute path)` to load the model.

Our driver.py provides easier usage. It can automatically process data, download models:
```bash
python driver.py  --input_path prot 
```

It can also download ten models trained by the ten-fold cross validation, and taking the average predicted value to achieve a more stable result.
```bash
python driver.py  --input_path prot --ten_average True 
```

To predict the output for all files in a directory:
```bash
python driver.py  --input_path <path_to_input_directory> --input_type dir --ten_average True 
```

## Reference

Kang, K., Wang, L., & Song, C. (2023). [ProtRAP: Predicting Lipid Accessibility Together with Solvent Accessibility of Proteins in One Run](https://doi.org/10.1021/acs.jcim.2c01235). _Journal of chemical information and modeling_.
