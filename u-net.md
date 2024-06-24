# U-Net
### Convolutional Networks for Biomedical Image Segmentation (2015)

one of the most influential papers in segmentation

### Abstract
- contracting path: captures context
- expanding path: precise localization
-----
### Introduction
- typical use of CNN is on classification tasks
- many visual tasks requires **localization**, which means that class label is supposed to be assigned to each pixel
- Each pixel is classified or mapped to certain labels
- Ciresan et al. used a sliding window approach and classified objects on a single pixel level, very expensive computational cost
  
![Ciresan Overview](./ciresan_overview.png)

#### Two Benefits of Ciresan's model
1. U-Net can localize
2. Training data in terms of patches is much larger than the number of training images

#### Two Benefits of Ciresan's model
1. U-Net can localize
2. Training data in terms of patches is much larger than the number of training images
#### Two rawbacks of Ciresan's model
1. Slow network, lot of redundancy due to overlapping patches
2. Trade-off between localization accracy and use of context (smaller the window, more localization, but less context)
-----
### Architecture

![Architecture of U-Net](./unet.png)
1. Contraction path
   - Input is a grayscaled 572 x 572 image
2. Expansion path
   - Output has two layers of 388 x 388, because it has two classes (size is reduced due to the input padding)
3. Skip Connections