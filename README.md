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
### SpatialRGPT-Bench

| Model                  | Height | Width | Vertical Distance | Horizontal Distance | Direct Distance | Average |
| :------------------ | -----: | ----: | ----------------: | ----------------: | ------------: | ------: |
| LLaVA-1.5-7B           |  10.53 | 15.04 |             16.98 |               17.21 |           13.51 |   14.49 |
| GPT-4.1                | 61.65 | 36.84 |              2.83 |                9.02 |           19.59 |   27.10 |
| SpatialRGPT            | ***63.61*** | ***48.12*** |             50.94 |               49.18 |           33.78 |   48.60 |       |        |       |                   |                     |                 |         |
| SpatialGeo |  58.65 | 41.35 |           ***56.60*** |           59.02 |       ***48.65*** | ***52.49*** |

#### Examples From SpatialRGPT-Bench
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

### SpatialScore
| Dataset             | LLaVA-1.5-7B | SpatialGeo-SA($\star$) | SpatialGeo-HA($\star$) |
| :------------------ | -----------: | ---------------------: | ---------------------: |
| QSpatial-Plus       |        38.61 |                  44.55 |              ***58.42*** |
| QSpatial-ScanNet    |        47.65 |                  50.59 |              ***54.12*** |
| SpatialBench        |    ***53.45*** |                  51.15 |              ***53.45*** |
| VSR-ZeroShot        |    ***70.13*** |                  69.31 |                  68.66 |
| SpatialSense        |        60.24 |              ***63.20*** |                  63.11 |
| RealWorldQA         |        54.38 |              ***57.25*** |                  55.69 |
| VGBench             |        31.79 |                  36.50 |              ***37.54*** |
| ***Average***         |        50.89 |                  53.22 |              ***55.86*** |

#### Examples From SpatialScore
<p align="center">
  <img src="READMEimages/spatialscore1.png" width="80%"/>
</p>
<p align="center">
  <img src="READMEimages/spatialscore2.png" width="80%"/>
</p>
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

## General VQA Benchmarks
We further test on ***general VQA benchmarks*** to evaluate whether the models overfit to the spatial tasks, and employ their respective evaluation criteria.
| Dataset                | LLaVA-1.5-7B | SpatialGeo-SA($\star$) | SpatialGeo-HA($\star$) |
| :--------------------- | -----------: | ---------------------: | ---------------------: |
| POPE (random)          | ***87.3***     | ***87.3***               | 86.7                   |
| POPE (popular)         | ***86.1***     | 86.0                   | 85.1                   |
| POPE (adversarial)     | 84.2         | ***84.5***               | 83.6                   |
| MM-Vet                 | ***31.1***     | 30.9                   | 30.9                   |
| MME                    | ***1504***     | 1464                   | 1470                   |
| MMVP                   | 41           | 31                     | ***42***                 |
| BLINK (relative_depth)           | 54.64        | 68.04                  | ***73.20***              |
____________________________________________________________________

#### Examples of Real World Photography
<p align="center">
  <img src="READMEimages/realworld.png" width="80%"/>
</p>