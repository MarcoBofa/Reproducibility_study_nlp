HyperLex is a gold standard resource for the evaluation of models that learn
the meaning of words and concepts. It provides a way of measuring how well
semantic models capture graded or soft lexical entailment (also known as
the type-of, is-a, or hypernymy-hyponymy relation) rather than semantic
similarity or relatedness. HyperLex comprises 2616 concept pairs in total
annotated for the relation of graded lexical entailment: 2163 noun pairs
and 453 verb pairs. Each pair of concepts (X,Y) is annotated according to
the question "To what degree is X a type of Y?"
:
A detailed description of graded lexical entailment and the data set, including
how data was collected and annotated can be found in the following publication:

HyperLex: A Large-Scale Evaluation of Graded Lexical Entailment. 2016. Ivan
Vulić, Daniela Gerz, Douwe Kiela, Felix Hill, and Anna Korhonen. (currently
on arXiv)

PLEASE CITE THIS PUBLICATION IF USING HYPERLEX IN YOUR RESEARCH

If you are unsure about certain aspects of the data set and graded lexical
entailment, as well as methodology used, we refer you to the aforementioned
publication. Otherwise feel free to send an e-mail to Ivan Vulić:
iv250@cam.ac.uk


################################################################################
# Files #

- ./HyperLex-All.txt: The main data set (all 2616 concept pairs)

- ./Nouns-Verbs/HyperLex-Nouns.txt: The noun subset of the entire data set
(2163 pairs)
- ./Nouns-Verbs/HyperLex-Verbs.txt: The verb subset of the entire data set
(453 pairs)


We also provide two standard splits (random and lexical) into training,
development, and test sets for parameter tuning and supervised learning:
 - ./Splits/Random/*
 - ./Splits/Lexical/*
The exact number of pairs in each subset for each split is available in
the article.


################################################################################
# Data Format #

The format is the same in each file provided. It is a whitespace separated
plaintext file, where rows correspond to concept pairs and columns correspond
to properties of each pair.


# WORD1: The first concept in the pair.

# WORD2: The second concept in the pair. Note that, unlike with similarity
or relatedness, the order is important as graded lexical entailment is an
asymmetric relation (i.e., the score for each pair (X,Y) answers the question
"To what degree is X a type of Y?")

# POS: The part-of-speech tag. N is for nouns, V is for verbs (determined
by occurrence in the POS-tagged British National Corpus). Only pairs of
matching POS are included in HyperLex.

# TYPE: The lexical relation of the pair according to WordNet. Possible values:
hyp-N, syn, ant, cohyp, mero, no-rel, rhyp-N (see the article for more details)

# AVG_SCORE: The HyperLex graded lexical entailment rating in the original
interval [0,6].

# AVG_SCORE_0_10 The graded lexical entailment ratings mapped from the interval
[0,6] to the interval [0,10] to match other datasets (e.g., SimLex-999,
WordSim-353, SimVerb-3500).

# STD Standard deviation of ratings (using original ratings in the interval
[0,6]. Low values indicate good agreement between the 10+ annotators on the
graded LE value AVG_SCORE. Higher scores indicate less certainty.

# SCORES..  All individual ratings in the interval [0,6] collected for the
concept pair from accepted annotators. Every pair has at least 10 ratings,
some pairs have more ratings.


Additional information about concepts (e.g., concreteness scores, association
scores) potentially supporting more analyses may be extracted from WordNet
or USF annotation data available here:
http://w3.usf.edu/FreeAssociation/AppendixA/index.html

