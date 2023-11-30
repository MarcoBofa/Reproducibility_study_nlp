# Reproducibility_study_nlp

Bonafini Marco - Bandera Calliope

In this project we reproduced the results of the paper "No clues, good clues: Out of context Lexical Relation Classification" [link](https://aclanthology.org/2023.acl-long.308). This research focuses on the use of pre‐trained language models (PTLMs) for lexical relation tasks in Natural Language Processing (NLP).

# Lexical Relation Classification and graded Lexical Entailment

## "No clues, good clues: Out of context Lexical Relation Classification"

This repository contains the datasets, scripts and notebooks needeed to reproduce the results in the paper: [_No clues, good clues: Out of context Lexical Relation Classification_](https://aclanthology.org/2023.acl-long.308). It contains:

- datasets: all datasets used in the paper. The folder `datasets/hyperlex-original` contains the original Hyperlex dataset. We use a modified version of this dataset (explained in Appendix A of the paper), `datasets/hyperlex`. It also contains the SoTA results in the literature (`datasets/sotas_results_literature/`) for a better visualization of the results.
- scripts: needed scripts to run the experiments.
- launchers: notebooks to launch the above scripts in Google Colab, and to collect the results.

### **Section 5.1 in the paper: Run the experiments**

To run the experiments in Section 5.1, following scripts are needed:

- `scripts/lrc_train_evaluate.py`: to run Lexical Relation Classification experiments with templates T1-T4.
- `scripts/masked_lrc_train_evaluate.py`: to run Lexical Relation Classification experiments with templates TM1-TM3.
- `scripts/gradedLE_train_evaluate.py`: to run graded Lexical Entailment experiments with templates T1-T4.
- `scripts/masked_gradedLE_train_evaluate.py`: to run graded Lexical Entailment experiments with templates TM1-TM3.

The scripts can be easly launched in Google Colab using the notebooks (upload one of them to your Colab and run all cells):

- `notebooks/lrc_tr-ev_launch.ipynb`
- `notebooks/masked_lrc_tr-ev_launch.ipynb`
- `notebooks/gradedLE_tr-ev_launch.ipynb`
- `notebooks/masked_gradedLE_tr-ev_launch.ipynb`

Usage examples of the scripts can be consulted in the regarding notebooks.

The launcher notebook has been updated to automatically execute experiments across all models, datasets, and templates without requiring manual changes. The only manual step involved is the gathering of results, which are saved in the Google Colab runtime, to run error analysis and visualization.

The script produces two results files:

- A csv file: it contains one line for each pair of words (source and target words) and the relation to predict in the test dataset. Each line is formed by: the predicted/real label relation id (number from $0$ to $K-1$, where $K$ is the number of the relations in the dataset) and the label relation name; tokenization of the source/target words and number of tokens; probability of each label; number of synsets of the source/target words; the entropy of the probability distribution of the labels.
- A txt file: the file contains $3$ lines. The first line is the printed version of a python dictionary with the value of all script parameters; the second line is the initial run date; and the thrid, the printed classification report of the function `classification_report` in the `sklearn` package.

The `masked_lrc_train_evaluate.py` script is used in a similar way removing the `test-templates` parameter. It also produces two files, adding the one token verbalization for the relation (predicted and real) and the top $2K$ tokens with the highest probabilities to fill the mask token.

The ''graded'' versions of the scripts, `gradedLE_train_evaluate.py` and `masked_gradedLE_train_evaluate.py`, follow a similar rationale than the regarding non-graded versions, with the following differences:

- the csv output files contains the human grades (instead of the predicted and real classification) and it is also added the produced logits by the model;
- the txt output files contains the Spearman correlations and the regression coefficients of the needed linear regresion to calculate the hyponym grade;
- it also produces a `.val` file with the logits and probabilities of each relation label for the pairs in the validation dataset.

### **Section 5.1 in the paper: Collect the results of the experiments**

Once the scripts are run, you get a set of results files like the ones in the `results` folder (due to space limitations, our results can be downloaded running the `results_processing.ypnb` notebook). To process these results files, it can be used the notebook `results_processing.ipynb` to collect all the results.

The section on gradedLE and the section on LRC that use the CogALexV dataset unfortunately do not work.

### **Section 5.2 in the paper: Error analysis**

This part is included just for completeness but does not actually work.

We tried to perform an error analysis of the results over the CogALexV and EVALution datasets. For this error analysis the metadata files are needed:

- For CogAlexV: `datasests/CogALexV/relata_metadata.txt` and `datasests/CogALexV/pairs_metadata.txt`.
- For EVALution: `datasests/EVALution/EVALUTION_RELATA.txt` and `datasests/EVALution/EVALUTION_RELATIONS.txt`.

These files contains the POS, domain, prototipically value needed to perform the error analysis.

This parte uses the notebook `error_analysis.ipynb`.

## **Summary of the results**

We report the results for RoBERTa large model trained with the different templates in the paper.

K&+N, BLESS, EVALution and ROOT9 results in terms of the weighted f1-score weighted by the support of the labels

| Model       | K&+N                 | BLESS                | EVALution            | ROOT9                |
| ----------- | -------------------- | -------------------- | -------------------- | -------------------- |
| RoBERTa/T1  | 0.989                | **0.954**            | **0.764**            | **0.936**            |
| RoBERTa/T2  | 0.989                | **0.955**            | **0.757**            | **0.936**            |
| RoBERTa/T3  | 0.989                | <ins>**0.956**</ins> | <ins>**0.771**</ins> | <ins>**0.937**</ins> |
| RoBERTa/T4  | 0.312                | 0.133                | 0.087                | **0.934**            |
| RoBERTa/TM1 | 0.988                | 0.947                | **0.761**            | **0.936**            |
| RoBERTa/TM2 | 0.988                | 0.946                | **0.764**            | **0.928**            |
| RoBERTa/TM3 | 0.985                | **0.951**            | **0.746**            | **0.926**            |
| LexNET      | 0.985                | 0.893                | 0.600                | 0.813                |
| KEML        | <ins>**0.993**</ins> | 0.944                | 0.660                | 0.878                |
| SphereRE    | 0.990                | 0.938                | 0.620                | 0.861                |
| RelBERT     | 0.949                | 0.921                | 0.701                | 0.910                |

CogAlexV results in terms of the f1-score for all relations and the weighted f1-score by the support of the labels

| Model       | ant                  | hyp                  | part                 | syn                  | all                  |
| ----------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| RoBERTa/T1  | **0.873**            | **0.703**            | **0.752**            | **0.604**            | **0.743**            |
| RoBERTa/T2  | **0.863**            | **0.682**            | **0.745**            | 0.584                | **0.728**            |
| RoBERTa/T3  | <ins>**0.884**</ins> | **0.718**            | **0.784**            | <ins>**0.629**</ins> | <ins>**0.762**</ins> |
| RoBERTa/T4  | 0.237                | 0.004                | 0.165                | 0.085                | 0.119                |
| RoBERTa/TM1 | **0.880**            | **0.709**            | **0.773**            | **0.599**            | **0.750**            |
| RoBERTa/TM2 | **0.871**            | <ins>**0.723**</ins> | <ins>**0.787**</ins> | **0.621**            | **0.758**            |
| RoBERTa/TM3 | **0.871**            | **0.718**            | **0.787**            | **0.616**            | **0.756**            |
| LexNET      | 0.425                | 0.526                | 0.493                | 0.297                | 0.445                |
| SphereRE    | 0.479                | 0.538                | 0.539                | 0.286                | 0.471                |
| KEML        | 0.492                | 0.547                | 0.652                | 0.292                | 0.500                |
| RelBert     | 0.794                | 0.616                | 0.702                | 0.505                | 0.664                |

Graded LE results over Hyperlex dataset in terms of the Spearman correlation for all pairs words and restricted to noun and verb pairs:

| Model       | random-split                                                   | lexical-split                                       |
| ----------- | -------------------------------------------------------------- | --------------------------------------------------- |
| RoBERTa/T1  | **0.741**/**0.753**/**0.584**                                  | **0.755**/**0.788**/**0.532**                       |
| RoBERTa/T2  | 0.152/0.170/0.030                                              | 0.287/0.350/0.063                                   |
| RoBERTa/T3  | 0.774/0.790/0.631                                              | **0.669**/**0.690**/**0.516**                       |
| RoBERTa/TM1 | <ins>**0.828**</ins>/<ins>**0.839**</ins>/<ins>**0.716**</ins> | **0.789**/<ins>**0.837**</ins>/**0.612**            |
| RoBERTa/TM2 | **0.749**/**0.761**/**0.646**                                  | **0.654**/**0.705**/**0.417**                       |
| RoBERTa/TM3 | **0.814**/**0.830**/**0.683**                                  | <ins>**0.794**</ins>/**0.828**/<ins>**0.656**</ins> |
| LEAR        | 0.686/0.710/------                                             | 0.174/------/------                                 |
| SDNS        | 0.692/------/------                                            | 0.544/------/------                                 |
| GLEN        | 0.520/------/------                                            | 0.481/------/------                                 |
| POSTLE      | 0.686/------/------                                            | ------/0.600/------                                 |
| LexSub      | 0.533/------/------                                            | ------/------/------                                |
| HF          | 0.690/------/------                                            | ------/------/-----                                 |

## **Citation**

To cite the paper

```
 @inproceedings{pitarch-etal-2023-clues,
    title = "No clues good clues: out of context Lexical Relation Classification",
    author = "Pitarch, Lucia  and
      Bernad, Jordi  and
      Dranca, Lacramioara  and
      Bobed Lisbona, Carlos  and
      Gracia, Jorge",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.308",
    pages = "5607--5625",
}
```
