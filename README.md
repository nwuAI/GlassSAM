***\*GlassSAM: A SAM-Based Multimodal Framework for Automatic Glass Segmentation\****

SAIL (Statistic Analysis And Intelligent Learning) Lab of NWU

We provide related codes and configuration files to reproduce the "GlassSAM: A SAM-Based Multimodal Framework for Automatic Glass Segmentation"



***\*Abstract:\****

Glass segmentation remains highly challenging due to the reflective and transmissive physical properties of glass, which often cause severe ambiguity and confusion between glass regions and surrounding objects in RGB images. Existing methods are often tailored to specific tasks or modality combinations, which limits their flexibility across different multimodal glass segmentation settings. To address these challenges, we propose GlassSAM, a SAM-based multimodal framework for glass segmentation. GlassSAM enhances the segmentation capability of SAM on RGB images by incorporating an auxiliary modality, such as depth, thermal, or polarization data. The proposed framework consists of two key components: a Control and Prompts Auto-Generation (CPAG) module that automatically generates control features as well as mask and point prompts, eliminating the need for manual interaction in SAM, and a Feature Controlled Encoder-Decoder (FCED) module that regulates the SAM encoder and decoder. Extensive experiments on public glass segmentation benchmarks demonstrate that GlassSAM achieves state-of-the-art performance.



***\*Example images：\****

<p align="center">
  <img src="./img/RGBP.png" alt="Image">
</p>


<p align="center">
  <img src="./img/RGBD.png" alt="Image">
</p>
<p align="center">
  <img src="./img/RGBT.png" alt="Image">
</p>

***\*Train the model:\****

```python
python train_glasssam.py
```



***\*Inference Dataset (Please load pre-training weights in advance):\****

```python
python test_glasssam.py
```



***\*Predict maps:\****

Link: 通过网盘分享的文件：predict maps
链接: https://pan.baidu.com/s/17oe3aRgsl7BGDISIwDtPAA?pwd=glas 

Extract code: glas



***\*Dataset:\****

rgbd_glass_Link: [OneDrive](https://portland-my.sharepoint.com/personal/jiayinlin5-c_my_cityu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjiayinlin5-c_my_cityu_edu_hk%2FDocuments%2FAAAI2025-RGBDGlass%2Fglass_depth_dataset.zip&parent=%2Fpersonal%2Fjiayinlin5-c_my_cityu_edu_hk%2FDocuments%2FAAAI2025-RGBDGlass&ga=1) 

rgbp_glass_link:[CVPR2022_PGSNet](https://mhaiyang.github.io/CVPR2022_PGSNet/)

rgbt_glass_link:[RGB-T-glass.zip - Google 雲端硬碟](https://drive.google.com/file/d/1g42XPaudslnIo1oR59FKQV9rVFOTqu_w/view)



***\*GlassSAM pretrained weights:\****

Link: 通过网盘分享的文件：weights
链接: https://pan.baidu.com/s/1RGzxqYIu7ODKXhVe49Snpg?pwd=glas  

Extract code: glas



***\*Environments:\****

```python
environment.yml
```

