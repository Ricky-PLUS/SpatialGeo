SpatialVLM: 基于 LLAVA 的空间型多模态视觉语言模型

环境配置：
推理：
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

训练：
pip install -e ".[train]"
<!-- pip install flash-attn --no-build-isolation -->
这种方法非常难下载，直接下载flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl（https://github.com/Dao-AILab/flash-attention/releases）
然后 pip install flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

在llava.eval下，可以找到benchmark

在spatialLLava终端下，先运行镜像export HF_ENDPOINT=https://hf-mirror.com
然后运行python -m llava.serve.cli 进行推理（要先下载llava模型参数和clip参数，但是我在a800超算平台已经下载好了）

moge一阶段训练脚本，在scripts.v1_5.pretrainmoge.sh