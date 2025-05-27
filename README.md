# XFinBench

This repository contains the code and data for the paper "XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning". Our work has been accepted to ACL 2025 Findings.

## About XFinBench

**XFinBench**, a novel benchmark designed to evaluate LLM's ability in solving comple**X**, knowledge-intensive **Fin**ancial problems across diverse graduate-level topics with multi-modal context. We identify five core capabilities of LLMs using XFinBench, _i.e_, _Terminology Understanding_ (TU), _Temporal Reasoning_ (TR), _Future Forecasting_ (FF), _Scenario Planning_ (SP), and _Numerical Modelling_ (NM).

## Dataset

All the examples in XFinBench were divided into two subsets: validation and test.

* validation: 1,000 examples used for model development, validation, or for those with limited computing resources.
* test: 3,235 examples for standard evaluation. Notably, the ground truthes and gold finance terms for test set will NOT be publicly released.

Data structure of examples in XFinBench is
```
id [str]: Unique id for each example.
task [str]: bool, mcq and calcu.
question [str]: the question in the example.
choice [str]: candidate choices if task is mcq.
ground_truth [str]: correct answer to the question. Masked in the test set.
figure [str]: figure name if visual-context required.
fin_capability [str]: TU, TR, FF, SP and NM.
gold_fin_term_id [int]: finance term id as ground truth of related background. Masked in the test set.
```

Data structure of terms in knowledge bank is
```
id [str]: Unique id for each finance term.
term_name [str]: the full name of terms.
term_definition [str]: the definition of terms.
```

## Code

Codes for dataset construction are in ```./construct_data``` folder. ```Generate_then_verify.py``` and ```QA_deduplication_bool.py``` correspond to _GPT-4o Enchanced Annotation_ section in our paper.

Codes for model evaluation are in ```./evaluate``` folder. Please change the experiment setting in ```./script/run_evaluate.sh``` and run the following command for evaluation:
```
bash ./script/run_evaluate.sh
```
