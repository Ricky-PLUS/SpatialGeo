<p align="center">
  <img src="READMEimages/SpatialGeo.png" width="15%"/>
</p>

# SpatialGeo: Boosting Spatial Reasoning in Multimodal LLMs via Geometry-Semantics Fusion
______________________________________________________________________

## Install
Run LLaVA on Linux.
1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/Ricky-PLUS/SpatialGeo.git
cd SpatialGeo/SpatialGeo
```

2. Install Package
```Shell
conda create -n mogellava python=3.10 -y
conda activate mogellava
pip install --upgrade pip 
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Train
SpatialGeo training consists of two stages: (1) feature alignment stage; (2) visual instruction tuning stage. SpatialGeo is trained on 8 A800 GPUs with 80GB memory.

### Stage-1
The training script is SpatialGeo/scripts/v1_5/pretrainmoge.sh
Download annotations and images from the following link (https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main).

### Stage-2
The training script is SpatialGeo/scripts/v1_5/finetunemoge.sh
Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/Ricky159/SpatialGeo), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
- Please download the original image from [OpenImagesV4] (https://storage.googleapis.com/openimages/web/download_v4.html) based on llava_v1_5_mix665k.json.

## Inferencing
Run the following command to inference a single image:
```Shell
python -m llava.serve.cli \
    --model-path "your_model_path" \
    --image-file "image_path" \
    --load-4bit
```
Here are some examples of inferencing:
<p align="center">
  <strong>Real World Photography</strong><br>
  <img src="READMEimages/realworld.png" width="80%"/>
</p>
<br>  <!-- 增加一个空行 -->


## Evaluation

You can find some evaluation benchmark tests in the SpatialGeo/llava/eval folder.
Most of the testing processes are consistent with LLaVA1.5

## Related Projects

- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA)

# References
If you find this repository useful for your research, please cite the following work.
```
@inproceedings{guo2025spatialgeo,
  title={SpatialGeo: Boosting Spatial Reasoning in Multimodal LLMs via Geometry-Semantics Fusion},
  author={Guo, Jiajie and Zhu, Qingpeng and Zeng, Jin and Wu, Xiaolong and He, Changyong and Wang, Weida},
  booktitle={27th IEEE International Workshop on Multimedia Signal Processing},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```
