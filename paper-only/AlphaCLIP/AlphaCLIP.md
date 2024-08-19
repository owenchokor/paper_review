# Alpha-CLIP
### A CLIP model Focusing on whatever you want

## Introduction

1. Empowering CLIP with region awareness
Alpha-CLIP incorporates an additional alpha channel which does not change the image content and preserves the generalized performance

2. Region level image annotation
Previous CLIP models were trained by large datasets, not including fine-grained mask labels
- GLIP: Kosmos-2, release GRIT dataset
- All-Seeing project: Pseudolabeling pipeline
- SAM: trained on massive vision modality data with strong zero-shot ability
**Alpha-CLIP is built upon GRIT and SAM to propose a method for generating RGBA region-text pairs from grounding data**

3. CLIP in MLLM(Multimodal LLM)

Alpha-CLIP + LLM = image captioning and VQA tasks
- CLIP encoder as important feature in MLLM
- CLIP maps text and image in same vector space
- Kosmos-2 uses box-corner point to use millions of region caption data

ROI allign : GPT4ROI

## Method
1. RGBA region - text pair generation
Data - generating pipeline to create RGBA-region text pairs
Natural image + caption
GRIT dataset-> automatic box-text label
SAM-> high quality pseudo mask
2. Alpha - CLIP
**Model Structure** 
- additional alpha convolution layer to RGB convolution in CLIP
**Training Method**
- fixed encoder, trains only Alpha-CLIP image encoder
- Alpha channel input
- low lr
- data sampling method : $r_s = 0.1$, all alpha channels to 1

## Experiments
1. Zero-Shot in ImageNet-S
2. Zero-Shot REC(Referring Expression Comprehension): finding specific objects based on text
3. OVD(Open Vocabulary detection): detects new classes(no bbox), not used in training
4. MLLM

-----
## Insights
- METEOR : metric for evaluating generated text
  - better than BLEU or ROUGE, since it evaluats consistency and synonyms
  - Prec, Rec, F-Score
- CIDEr : image caption evaluation
  - TF-IDF : Term frequency Inverse Document Frequency : importance of each word
  - Cosine Similarity : Vector similarity
  - N-gram matching : continuous sequence
- Datasets: MS-COCO, Visual Genome(labeled 100k), YFCC100M(100M, low quality)
- CLIP loss : image - text softmax + text - image softmax
- Related works : ReCLIP (Region proposal + CLIP), MaskAdaptedCLIP