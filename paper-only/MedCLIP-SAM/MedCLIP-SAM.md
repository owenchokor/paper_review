# MedCLIP-SAM
### Bridging Text and Image Towards Universal Medical Image Segmentation

-----

### **Abstract**
MedCLIP-SAM is a medical image segmentation model that combines MedCLIP and SAM.
- important roles in modern clinical practice & studies
- **Goal**: Creating a low-label, accurate segmentation tool
- **Previous models** : CLIP(Contrastive Language-Image Pre-Training), SAM(Segment Anything Model)
- **New Suggestions**
   1. **DHN-NCE** : a new form of loss (Decoupled Hard Negative Noise Contrastive Es-timation)
   2. **gScoreCAM**: prompt for creating segmentation mask on SAM
   3. Zero-Shot and weakly supervised tasks
- **Achievements** : Segmentation tasks on various imaging tools (e.g. Breast Cancer, Brain Tumor MRI, Lung X-Ray)
  

### **Introduction**
1. CLIP(Contrastive Language-Image Pre-Training)
   - text - image align
   - image encoders and text encoders
   - similar entities aligned closely in the vector space
   - contrastive learning
2. MedSAM
   - Medical Image Segmentation in various domains
   - Uses un/weakly- supervised models to improve SAM
   - interaction with text prompt

MedCLIP-SAM leverages BiomedCLIP and SAM for text-prompt based interactive an universal medical image segmentation both in zero-shot and weakly supervised settings.


### **Materials & Method**
1. BiomedCLIP fine tuning with DHN-NCE loss 
   - Public MedPix dataset
   - Preprocessing: deleted special character, spaces, 20 less captions
2. zero-shot segmentation guided by text-prompts
   - gScoreCAM: provides visual saliency maps of text prompts in corresponding images for CLIP models (outperforms gradCAM in natural images, first use in medical images)
   - CRF filter: cost segmentation
   - zero-shot segmentation by pseudo - mask
   - weak supervised learning: Residual UNet by generated pseudo - masks
3. weakly supervised segmentation for potential label refinement