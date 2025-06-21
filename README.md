<p align="center">
  <img src="READMEimages/SpatialGeo.png" width="15%"/>
</p>

# SpatialGeo: Boosting Spatial Reasoning in Multimodal LLMs via Geometry-Semantics Fusion
______________________________________________________________________

***SpatialGeo*** enhances ***spatial reasoning*** in MLLMs based on the novel vision encoder generating spatial-aware visual embedding.

The overall architecture of ***SpatialGeo*** is shown in the figure below, which is composed of three major modules: 1) ***CLIP module*** with the CLIP encoder and its adapter to extract instance-level semantic features; 2) ***MoGe module*** with the MoGe encoder and its adapter to embed a mixture of geometry and semantic features; 3) ***LLM module*** with interleaved geometry and semantic embeddings together with text tokens as inputs to generate question answering.

<p align="center">
  <img src="READMEimages/structure.png" width="80%"/>
</p>

______________________________________________________________________

## Spatial VQA Datasets
We compare SpatialGeo with SOTA MLLMs on spatial VQA datasets, including ***SpatialRGPT-Bench*** and ***SpatialScore***.
### Examples From SpatialRGPT-Bench
<p align="center">
  <img src="READMEimages/rgpt1.png" width="80%"/>
</p>
<p align="center">
  <img src="READMEimages/rgpt2.png" width="80%"/>
</p>
<p align="center">
  <img src="READMEimages/rgpt3.png" width="80%"/>
</p>
<p align="center">
  <img src="READMEimages/rgpt4.png" width="80%"/>
</p>
<p align="center">
  <img src="READMEimages/rgpt5.png" width="80%"/>
</p>

______________________________________________________________________

### Examples From SpatialScore
We select different types of questions from ***SpatialScore*** for presentation.
<p align="center">
  <strong>Fig.1 Depth and Distance</strong><br>
  <img src="READMEimages/spatialscoreDepthanddistance.png" width="80%"/>
</p>    

<br>  <!-- 增加一个空行 -->

<p align="center">
  <strong>Fig.2 Object Localization</strong><br>
  <img src="READMEimages/spatialscoreObjectLocalization.png" width="80%"/>
</p>

<br>  <!-- 增加一个空行 -->

<p align="center">
  <img src="READMEimages/spatialscore3.png" width="80%"/>
</p>
<p align="center">
  <img src="READMEimages/spatialscore4.png" width="80%"/>
</p>
<p align="center">
  <img src="READMEimages/spatialscore5.png" width="80%"/>
</p>

______________________________________________________________________

### Examples of Real World Photography
<p align="center">
  <img src="READMEimages/realworld.png" width="80%"/>
</p>