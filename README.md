
We modified this repo to reproduce the experiments for the paper <a href="https://openreview.net/pdf?id=9hjVoPWPnh">Machine Unlearning for Image-to-Image Generative Models</a> (ICLR 2024).

You shall run the code in the current dir: `i2i_mage/`

# Setup


### CKPT
Download the pretrained model from [G-Drive](https://drive.google.com/file/d/1Q6tbt3vF0bSrv5sPrjpFu8ksG3vTsVX2/view). 

Beside, please download the pretrained VQ-GAN model [G-Drive](https://drive.google.com/file/d/13S_unB87n6KKuuMdyMnyExW0G1kplTbP/view).

You can download them and put them under `pretrained`.

# Usage

### **Ours Approach**
Dataset options:
- --retainset_ratio: the ratio of the number of **REAL** retain samples [NO PROXY] vs. the number of forget samples. 
- --use_coco: use COCO as the proxy of retain set or not
  - if not: will use the remaining 800 classess from imagenet-1k as the proxy retain set
- --opendata_to_forget_ratio: the ratio of the number of **TOTAL** retain samples [REAL+PROXY] vs. the number of forget samples.
  - Note if, not equal `1`, will oversample to make them balanced
- --num_image_per_class: number of images per class for the forget set.

|Approach|--retainset_ratio|--use_coco|--opendata_to_forget_ratio|
|-|-|-|-|
|  **Use REAL retain ONLY**  | >0   |  0  |  0  | 
|  **Use COCO only**  |  0  |  1  |  >0  |
|  **Use rest 800 classes from IN-1K only**  |  0  |  0  |  >0  |
|  **Use COCO +REAL retain**  |  >0  |  1  |  >0  |
|  **Use rest 800 classes from IN-1K +REAL retain**  |  >0  |  0  |  >0  |


- example:
   - `python -m torch.distributed.launch --nproc_per_node=4 main_unlearn.py --resume pretrained/mage-vitb-1600.pth --blr 0.0001 --forget_alpha 0.2`

### **Run other baseline methods **
Define `loss_type` in the following code
`python -m torch.distributed.launch --nproc_per_node=4 main_baseline.py --resume pretrained/mage-vitb-1600.pth --blr 0.0001 --forget_alpha 0.2 --loss_type $LOSS_TYPE`

    - Retain Label: $LOSS_TYPE == 'learn_others'
    - MAX Loss: $LOSS_TYPE =='max_full_model_loss'
    - Random Encoder: $LOSS_TYPE == 'encoder_noise'
    - Noise Label: $LOSS_TYPE == 'full_model_noise'
- Example of MAX Loss:
  - `python -m torch.distributed.launch --nproc_per_node=4 main_baseline.py --resume pretrained/mage-vitb-1600.pth --blr 0.0001 --forget_alpha 0.2 --loss_type full_model_noise`

# Test
### **Generate images**

**Inpaint**

CUDA_VISIBLE_DEVICES=0 python testpaint.py --task inpaint --dataset img --ckpt_path $CKPT_PATH

**Outpaint**

CUDA_VISIBLE_DEVICES=1 python testpaint.py --task outpaint --dataset img --ckpt_path $CKPT_PATH


- Example: test the model before unlearning:

```
CUDA_VISIBLE_DEVICES=0 python testpaint.py --task outpaint --dataset img --ckpt_path pretrained/mage-vitb-1600.pth
CUDA_VISIBLE_DEVICES=1 python testpaint.py --task inpaint --dataset img --ckpt_path pretrained/mage-vitb-1600.pth
```

### **Compute FID and IS**
Note: the installtion of FID:


`python eval.py --dst $PATH_OF_GENERATED_IMG_FOLDERS --task inpaint`

`python eval.py --dst $PATH_OF_GENERATED_IMG_FOLDERS --task outpaint`

`$PATH_OF_GENERATED_IMG_FOLDERS` is the path of the tested checkpoint by `$CKPT_PATH.replace('.pth', '-inpaint/')` or `$CKPT_PATH.replace('.pth', '-outpaint/')`

- Example: test the model before unlearning:

	- `python eval.py --dst pretrained/mage-vitb-1600-inpaint/ --task inpaint`
	- `python eval.py --dst pretrained/mage-vitb-1600-outpaint/ --task outpaint`
- Results: The results for FID and IS score will be stored at `fid_is_eval_{MODE}.csv`， where `{MODE}` is `inpaint` or `outpaint`.

### **Computer CLIP score**

Environment preparation:
```
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
mv open_clip_pytorch_model.bin pretrained/open_clip_vit_h_14_laion2b_s32b_b79k.bin
```

Prepare the clip embedding of original iamges:

`python clip_embed.py --img_folder original --task inpaint`

`python clip_embed.py --img_folder original --task outpaint`


Run the generated images:

`python clip_embed.py --img_folder $PATH_OF_GENERATED_IMG_FOLDERS --task inpaint`

`python clip_embed.py --img_folder $PATH_OF_GENERATED_IMG_FOLDERS --task outpaint`

- Example: test the model before unlearning:

	- `python clip_embed.py --img_folder pretrained/mage-vitb-1600-inpaint/ --task inpaint`
	- `python clip_embed.py --img_folder pretrained/mage-vitb-1600-outpaint/ --task outpaint`
- Results: The results for CLIP score will be stored at `clip_cosine_{MODE}.csv`， where `{MODE}` is `inpaint` or `outpaint`.


### **T-SNE**

After CLIP, run T-SNE with embedding

`python tsne_visual.py --ckpt_folder $PATH_OF_GENERATED_IMG_FOLDERS`












