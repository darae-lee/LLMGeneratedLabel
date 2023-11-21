# LLMGeneratedLabel

Generate labels based on the context and classification criterion(type) below. And evaluate the performance.


|         |      0     |          1          |         2         |           3          |
|:-------:|:----------:|:-------------------:|:-----------------:|:--------------------:|
| context | no context | general description | short description | detailed description |

|      |   0   |          1         |     2    |            3           | 4                  |
|:----:|:-----:|:------------------:|:--------:|:----------------------:|:----------------------:|
| type | basic | population density | building | peripheral environment | development degree |


## Generate LLM Label

```
python 1-get_label.py [--type TYPE] [--context CONTEXT]
                      [--model MODEL-DIR] [--processor PROCESSOR-DIR]
                      [--d-dir DATA-DIR] [--o-dir OUTPUT-DIR]
```

Creat a model output file <code>{o-dir}/output_context{context}_type{type}.csv</code> of [image, type], and a hard label file <code>{o-dir}/label_context{context}_type{type}.csv</code> of [image, urban, natural, environment].

MODEL-DIR and PROCESSOR-DIR contains InstructBlip model and processor, respectively. It generates labels for all 'png' images in DATA-DIR.


## Train the satellite imagery-based economic scale scoring model

```
python 2-proxy_pretrain.py [--l-path LABEL-PATH] [--i-dir IMAGE-DIR]
                           [--o-dir OUTPUT-DIR]
                           [--batch-size BACTH-SIZE] [--epoch EPOCH]
```

Train the satellite imagery-based economic scale scoring model using images(IMAGE-DIR) and labels(LABEL-PATH). Model's checkpoint is saved in OUTPUT-DIR as <code>checkpoint_context{context}_type{type}.ckpt</code>. It automatically make and use the train and test data at a ratio of 8:2.

The satellite imagery-based economic scale scoring model and data loader should be located on <code>model/model.py</code>, <code>model/dataloader.py</code>.


## Evaluate

```
python 3-check_corr.py [--cp-dir CHECKPOINT-DIR] [--meta-dir META-DIR]
                       [--album-dir ALBUM-DIR] [--csv-dir CSV-DIR]
                       [--o-dir OUTPUT-DIR]
```
Evaluate by Pearson correlation coefficient with the actual number of population.
CHECKPOINT-DIR is a directory containing checkpoints of the model to be evaluated. The checkpoint name should be in <code>checkpoint_context{context}_type{type}.ckpt</code> format.

It generates Pearson correlation coefficient files named <code>{output_dir}/context{context}_type{type}.csv</code> and <code>{output_dir}/context{context}_type{label_type}_grid.csv</code>.


### Settings for evaluation

- Place the files below in META-DIR
  - <code>geographical_info.csv</code> of [,adm1,adm2,adm3]
  - <code>geographical_info_list.csv</code> of [,adm1,adm2,adm3list]
  - <code>population.csv</code> of [,adm1,adm2,adm3,pop]
  - adm1, adm2, and adm3 are geographical information units, and they are metropolitan city/Do(province), Si/Gun, Myeon/Eup/Dong, respectively.

- Place satellite imageries in ALBUM-DIR
  - Each image is located at ALBUM-DIR/adm1/adm2/adm3/.

- Place csv file containing 'geometry(latitude, longitude)' and population information.
  - <code>{adm1}.csv</code> of [,areaid,geometry,pop,adm1,adm2,adm3]

## Ensemble

```
python3.8 4-ensemble.py [--path PATH]
```

Ensemble the Pearson correlation coefficient files(generated in Evaluation) located in PATH, as the average value.

