# LDSGM
## Code for paper: A Label Dependence-aware Sequence Generation Model for Multi-level Implicit Discourse Relation Recognition (AAAI 2022)

### Main Dependecies and installation

pytorch 1.3.1

transformer 4.12.4

pytorch_pretrained_bert 0.6.2

### DataSet

Download the [PDTB2.0](https://www.seas.upenn.edu/~pdtb/) and process the data into the following format:

![](https://github.com/nlpersECJTU/LDSGM/sample.png)

and put it under /PDTB/Ji/data/

Download the pretrained model [roberta base](https://huggingface.co/roberta-base/tree/main),put it under /pretrained/roberta-base/

## Run

```py
python3 run.py
```

