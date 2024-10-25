# LA3D 
A lightweight adaptive anonymization for VAD (LA3D) that employs dynamic adjustment to enhance privacy protection. 

## Abstract (You can read the full paper in [arXiv](https://arxiv.org/abs/2410.18717))
Recent advancements in artificial intelligence promise ample potential in monitoring applications with surveillance cameras.  
However, concerns about privacy and model bias have made it challenging to utilize them in public. Although de-identification approaches have been proposed in the literature, aiming to achieve a certain level of anonymization, most of them employ deep learning models that are computationally demanding for real-time edge deployment. 
In this study, we revisit conventional anonymization solutions for privacy protection and real-time video anomaly detection (VAD) applications. We propose a novel lightweight adaptive anonymization for VAD (LA3D) that employs dynamic adjustment to enhance privacy protection. 
We evaluated the approaches on publicly available privacy and VAD data sets to examine the strengths and weaknesses of the different anonymization techniques and highlight the promising efficacy of our approach. 
Our experiment demonstrates that LA3D enables substantial improvement in the privacy anonymization capability without majorly degrading VAD efficacy.

## Code is coming soon!!!

## Performance on Privacy Attribute Detection  vs. Video Anomaly Detection

### Using PEL4VAD VAD Model
<img src="./results/PEL4VAD_ad_ucf_auc_pd_vispr_cmap_clf_compare_all.jpg" alt="PEL4VAD VAD on UCF Crime vs. PD on VISPR" title="PEL4VAD VAD on UCF Crime vs. PD on VISPR" width=100% height=100%>

### Using MGFN VAD Model
<img src="./results/MGFN_ad_ucf_auc_pd_vispr_cmap_clf_compare_all.jpg" alt="MGFN VAD on UCF Crime vs. PD on VISP" title="MGFN VAD on UCF Crime vs. PD on VISPR" width=100% height=100%>


## Examples: Anonymization Enhancement using our Adaptive Approach (_A)

### 1:RAW\_IMAGE, 2:BLACKENED, 3:BLACKENED_EDGED, 4:PIXELIZED_D2, 5:PIXELIZED_D4, 6:PIXELIZED_D8, 7:BLURRED
<img src="./results/vispr_anony_compare_all_part_1_im_s320_240_images_2017_17368641.jpg" alt="images_2017_17368641" title="" width=100% height=100%>

###  8:PIXELIZED_D2_A ($\alpha_b=0.5$), 9:PIXELIZED_D4_A ($\alpha_b=0.5$), 10:PIXELIZED_D8_A ($\alpha_b=0.5$), 11:PIXELIZED_A ($ismax=True$, $D_a=Z_b$), 12:BLURRED_A ($\alpha_b=0.5$), 13:BLURRED_A ($ismax=True$, $K_a=Z_b$)
<img src="./results/vispr_anony_compare_all_part_2_im_s320_240_images_2017_17368641.jpg" alt="images_2017_17368641" title="" width=100% height=100%>

### 1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A
<img src="./results/vispr_anony_compare_D4_adpative_im_s320_240_images_2017_40438231.jpg" alt="VSIPR_TEST_IMAGE_40438231" title="1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A" width=100% height=100%>
<img src="./results/vispr_anony_compare_D4_adpative_im_s320_240_images_2017_29920650.jpg" alt="VSIPR_TEST_IMAGE_29920650" title="1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A" width=100% height=100%>
<img src="./results/vispr_anony_compare_D4_adpative_im_s320_240_images_2017_31772060.jpg" alt="VSIPR_TEST_IMAGE_31772060" title="1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A" width=100% height=100%>
<img src="./results/vispr_anony_compare_D4_adpative_im_s320_240_images_2017_99544991.jpg" alt="VSIPR_TEST_IMAGE_99544991" title="1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A" width=100% height=100%>
<img src="./results/vispr_anony_compare_D4_adpative_im_s320_240_images_2017_14412647.jpg" alt="VSIPR_TEST_IMAGE_14412647" title="1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A" width=100% height=100%>
<img src="./results/vispr_anony_compare_D4_adpative_im_s320_240_images_2017_32563909.jpg" alt="VSIPR_TEST_IMAGE_3256390" title="1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A" width=100% height=100%>
<img src="./results/vispr_anony_compare_D4_adpative_im_s320_240_images_2017_50916691.jpg" alt="VSIPR_TEST_IMAGE_50916691" title="1: RAW_IMAGE, 2: PIXELIZED_D4, 3: PIXELIZED_D4_A, 4: BLURRED, 5: BLURRED_A" width=100% height=100%>

## BibTeX Citation

If you employ any part of the code, please kindly cite the following papers:
```
@article{asres2024la3d,
  title={Low-Latency Video Anonymization for Crowd Anomaly Detection: Privacy vs. Performance},
  author={Asres, Mulugeta Weldezgina and Jiao, Lei and Omlin, Christian Walter},
  journal={arXiv preprint arXiv:2410.18717},
  year={2024}
}
```
