General caveats
=====================================

Comparisons between force fields
.....................................
Absolute values (for epsilon or intermaps) between different force fields are not automatically comparable. It's certainly possible to regularize values from a force field such that they can be comparable, but off the bat we should not expect values to be comparable. In short, it is strongly recommended to run analysis pipelines using one specific force field rather than mixing and matching.


Force field limitations
...........................
Coarse-grained force fields are good at capturing certain effects, and less good at others. Their strengths and weaknesses vary between different models, but it's very important to recognize that FINCHES will not be able to offer predictive power for types of physical chemistry not well-captured by the underlying force field being used.


Lack of exclusivity in terms of interactions
................................................
An intermap provides a prediction of "which regions of my two proteins would likely be attractive for one another". It does NOT directly provide information describing how I might expect the bound state of the two proteins to be. This is because the intermaps do not incorporate long-range correlations; if there is a highly promiscous subregion with an IDR, the intermap may predict it to interact favourably with many other IDRs, but this does not mean that all of these interactions will be observed in practice. In other words, the intermap is a prediction of the "potential" for interaction, but not a prediction of the "actual" interaction.



