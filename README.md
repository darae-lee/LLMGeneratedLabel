# LLMGeneratedLabel

Generate labels based on the context and classification criterion(type) below. And evaluate the performance.


|         |      0     |          1          |         2         |           3          |
|:-------:|:----------:|:-------------------:|:-----------------:|:--------------------:|
| context | no context | general description | short description | detailed description |

|      |   0   |          1         |     2    |            3           | 4                  |
|:----:|:-----:|:------------------:|:--------:|:----------------------:|:----------------------:|
| type | basic | population density | building | peripheral environment | development degree |



## Generate LLM Label

<code> python 1-get_label.py --type {type} --context {context} </code>

Create a hard label file <code>label_context{context}_type{type}.csv</code> of [image_id, urban, natural, environment].


## Train the satellite imagery-based economic scale scoring model

<code> python 2-proxy_pretrain.py --type {type} --context {context} </code>

Train the model using <code>label_context{context}_type{type}.csv</code>. Model's checkpoint is saved in <code>./save_model</code>.


## Evaluate

<code> python 3-check_corr.py </code>

Evaluate by Pearson's correlation coefficient with the actual number of population. It uses checkpoints stored in <code>./save_model</code>


## Ensemble

<code> python 4-ensemble.py --grid {true/false} --type {types} --context {context} </code>

ex) <code> python 4-ensemble.py --grid false --type 0 2 --context 2 1 </code>

Enter the type and context you want to ensemble. In the above example, the label type0 & context2 and type2 & context1 are ensemble.
