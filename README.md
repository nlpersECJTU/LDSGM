# LDSGM
##  A Label Dependence-aware Sequence Generation Model for Multi-level Implicit Discourse Relation Recognition (AAAI 2022)

### Main Dependecies and installation

pytorch 1.3.1

transformer 4.12.4

pytorch_pretrained_bert 0.6.2

### DataSet

Download the [PDTB2.0](https://www.seas.upenn.edu/~pdtb/) ,divided into training set(train.txt),validation set(dev.txt),test set(test.txt) and process the data into the following format: ['Top-level label1', 'Second-level label1', 'connective1'] ||| ['Top-level label2', 'Second-level label2', 'connective2'] ||| arg1 ||| arg2 .
Note that an instance may be annotated with more than one relation type.

![](https://github.com/nlpersECJTU/LDSGM/blob/main/sample.png)

and put it under /PDTB/Ji/data/

Download the pretrained model [roberta base](https://huggingface.co/roberta-base/tree/main),put it under /pretrained/roberta-base/

## Run

```py
python3 run.py
```

