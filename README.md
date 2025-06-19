# SpatialGeo: Boosting Spatial Reasoning in Multimodal LLMs via Geometry-Semantics Fusion

![alt text](READMEimages/first.png)

***SpatialGeo*** enhances ***spatial reasoning*** in MLLMs based on the novel vision encoder generating spatial-aware visual embedding.

The overall architecture of ***SpatialGeo*** is shown in the figure below, which is composed of three major modules: 1) ***CLIP module*** with the CLIP encoder and its adapter to extract instance-level semantic features; 2) ***MoGe module*** with the MoGe encoder and its adapter to embed a mixture of geometry and semantic features; 3) ***LLM module*** with interleaved geometry and semantic embeddings together with text tokens as inputs to generate question answering.

![alt text](READMEimages/structure.png)

## Spatial VQA Datasets
We compare SpatialGeo with SOTA MLLMs on spatial VQA datasets, including ***SpatialRGPT-Bench*** which is the testing dataset in OSD, and ***SpatialScore*** to evaluate the generalization to unseen dataset.
### SpatialRGPT-Bench
We use $\alpha$ to denote removing the CLIP branch in the first-stage training, $\beta$ to denote random feature dropping for CLIP in the second stage, SA to denote single adapter using the last block in MoGe, and HA to denote hierarchical adapter.
| Model                  | Height | Width | Vertical Distance | Horizontal Distance | Direct Distance | Average |
| :------------------ | -----: | ----: | ----------------: | ----------------: | ------------: | ------: |
| LLaVA-1.5-7B           |  10.53 | 15.04 |             16.98 |               17.21 |           13.51 |   14.49 |
| LLaVA-OSD              |  54.14 | 34.59 |           ***56.60*** |               50.82 |           40.54 |   46.73 |
| GPT-4o                 |  18.80 | 10.53 |              4.72 |                5.74 |            2.03 |    8.41 |
| GPT-4V                 |  24.06 | 21.05 |              6.60 |                9.02 |            7.43 |   13.86 |
| GPT-4.1                | 61.65 | 36.84 |              2.83 |                9.02 |           19.59 |   27.10 |
| SpatialRGPT            | ***63.61*** | ***48.12*** |             50.94 |               49.18 |           33.78 |   48.60 |
| ***Variants of SpatialGeo***     |        |       |                   |                     |                 |         |
| SpatialGeo-SA          |  54.14 | 44.36 |             54.72 |               55.74 |           38.51 |   48.91 |
| SpatialGeo-SA ($\beta$) |  48.12 | 37.59 |           ***56.60*** |           ***63.93*** |       ***48.65*** | 50.47 |
| SpatialGeo-HA ($\alpha$, $\beta$) |  18.05 | 17.29 |             26.42 |               18.85 |           21.62 |   20.25 |
| ***Full model of SpatialGeo***           |        |       |                   |                     |                 |         |
| SpatialGeo-HA ($\beta$) |  58.65 | 41.35 |           ***56.60*** |           59.02 |       ***48.65*** | ***52.49*** |

### SpatialScore
| Dataset             | LLaVA-1.5-7B | SpatialGeo-SA($\beta$) | SpatialGeo-HA($\beta$) |
| :------------------ | -----------: | ---------------------: | ---------------------: |
| QSpatial-Plus       |        38.61 |                  44.55 |              ***58.42*** |
| QSpatial-ScanNet    |        47.65 |                  50.59 |              ***54.12*** |
| SpatialBench        |    ***53.45*** |                  51.15 |              ***53.45*** |
| VSR-ZeroShot        |    ***70.13*** |                  69.31 |                  68.66 |
| SpatialSense        |        60.24 |              ***63.20*** |                  63.11 |
| RealWorldQA         |        54.38 |              ***57.25*** |                  55.69 |
| VGBench             |        31.79 |                  36.50 |              ***37.54*** |
| ***Average***         |        50.89 |                  53.22 |              ***55.86*** |

## General VQA Benchmarks
We further test on ***general VQA benchmarks*** to evaluate whether the models overfit to the spatial tasks, and employ their respective evaluation criteria.
| Dataset                | LLaVA-1.5-7B | SpatialGeo-SA($\beta$) | SpatialGeo-HA($\beta$) |
| :--------------------- | -----------: | ---------------------: | ---------------------: |
| POPE (random)          | ***87.3***     | ***87.3***               | 86.7                   |
| POPE (popular)         | ***86.1***     | 86.0                   | 85.1                   |
| POPE (adversarial)     | 84.2         | ***84.5***               | 83.6                   |
| MM-Vet                 | ***31.1***     | 30.9                   | 30.9                   |
| MME                    | ***1504***     | 1464                   | 1470                   |
| MMVP                   | 41           | 31                     | ***42***                 |
| BLINK (relative_depth)           | 54.64        | 68.04                  | ***73.20***              |
