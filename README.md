# XFinBench

This repository contains the code and data for the paper "XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning". This responsitory is only used for double-blind reviewing. 

## About XFinBench

**XFinBench**, a novel benchmark designed to evaluate LLM's ability in solving comple**X**, knowledge-intensive **Fin**ancial problems across diverse graduate-level topics with multi-modal context. We identify five core capabilities of LLMs using XFinBench, _i.e_, _Terminology Understanding_ (TU), _Temporal Reasoning_ (TR), _Future Forecasting_ (FF), _Scenario Planning_ (SP), and _Numerical Modelling_ (NM).

## Dataset Usage

All the examples in XFinBench were divided into two subsets: validation and test.

* validation: 1,000 examples used for model development, validation, or for those with limited computing resources.
* test: 3,235 examples for standard evaluation. Notably, the ground truthes and gold finance terms for test set will NOT be publicly released.

Data structure of examples in XFinBench is
```
id: Unique id for each example.
task: bool, mcq and calcu.
question: the question in the example.
choice: candidate choices if task is mcq.
ground_truth: correct answer to the question. Masked in the test set.
figure: figure name if visual-context required.
fin_capability: TU, TR, FF, SP and NM.
gold_fin_term_id: finance term id as ground truth of related background. Masked in the test set.
```

Data structure of terms in knowledge bank is
```
id: Unique id for each finance term.
term_name: the full name of terms.
term_definition: the definition of terms.
```

## Code

We will update codes for dataset construction and experiments as soon as possible. Thank you for your patience!
