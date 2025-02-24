# Breast-Cancer-detection-using-Histopathology-Images
## BCI_dataset

### Directory Structure

BCI_dataset
│
├── HE
│ ├── train
│ └── test
│
└── IHC
├── train
└── test
### Statistics

Our BCI dataset contains 9746 images (4873 pairs), 3896 pairs for train and 977 for test.  
The naming convention for images is 'number_train/test_HER2level.png'. Note that the HER2 level represents which category of WSI the image came from, not the image itself.
The same pair of HE and IHC images have the same number.  

Note: The HER2 level indicates the category of the Whole Slide Image (WSI) from which the image originated, not the image itself.  
The same pair of HE and IHC images share the same number.  

### License  

This BCI Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation.  
Permission is granted to use the data given that you agree to our license terms below:  

1. Include a reference to the BCI Dataset in any work that utilizes the dataset. For research papers, cite our preferred publication as listed on our website. For other media, cite our preferred publication or link to the BCI website.  
2. The dataset or any derivative work cannot be used for commercial purposes, such as licensing, selling the data, or using the data for commercial gain.  
3. All rights not expressly granted are reserved by us.  

### Privacy  

All data is desensitized, and private information has been removed.  
For any privacy concerns, please contact us at:  
- shengjie.Liu@bupt.edu.cn  
- czhu@bupt.edu.cn  
- bupt.ai.cz@gmail.com  

### Citation  

If you use this dataset for your research, please cite our paper:  
**BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix**  

```bibtex
@article{liu2022bci,
  title={BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix},
  author={Liu, Shengjie and Zhu, Chuang and Xu, Feng and Jia, Xinyu and Shi, Zhongyue and Jin, Mulan},
  journal={arXiv preprint arXiv:2204.11425},
  year={2022}
}
