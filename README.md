
__Stratified cross-validation for multi-label classification__

One way how to evaluate the accuracy of machine learning models is via cross-validation.
When we are dealing with classification, we may want to use _stratified_ cross-validation,
which preserves the distribution of the classes in the whole data set in the individual folds.
However, common implementations of stratified cross-validation work only with a single label. 
This code performs stratified assignment of multi-label samples into folds, where the labels are all nominal.  


__Assignment objectives__

1. Preserve the distribution of individual class-values across folds (_1-way interaction_)
2. Preserve the distribution of _2-way interactions_ between individual class-values across folds
3. Preserve the distribution of _n-way interactions_ between individual class-values across folds, where _n_ is the count of labels  


__Literature review__

One way how to quickly extend stratified cross-validation into multi-label stratified cross-validation is by concatenating the class labels into a single label. And run the standard stratified cross-validation. This approach takes care of preserving the _n-way interactions_ listed above, but of nothing else.

Another approach is to maintain _1-way interactions_. 
This was done by [(Sechidis, 2011)][1]. And later on extended by [(Szymański, 2017)][2] to optimize both, _1-way_ and _2-way_ interactions.
We optimize all these three criteria at once. 


__Why bother?__

Stratified cross-validation generally improves (plain) cross-validation in the following aspects:

1. It makes sure that each class-value is present in the testing set. This is important for the evaluation of many performance measures.
2. It maintains the same class prior distribution across all the folds. This increases the measured testing accuracy and minimizes the variance of the testing accuracy.

 
__Solution__

We use Integer Linear Programming (ILP) to reach the optimal solution. Hence, the solution is not an approximation but is exact.
The disadvantage, in comparison to greedy solutions from [(Sechidis, 2011)][1] and [(Szymański, 2017)][2] is that the calculation is slow.
Hence, we provide pre-calculated assignments for 10-fold cross-validation for some common multi-label classification data sets at [Multi-Label Classification Dataset Repository](http://www.uco.es/kdis/mllresources/).      


__Acknowledgements__
1. The data are from [Multi-Label Classification Dataset Repository][3] by Mojano et al.
2. The first published article about stratified multi-label cross-validation is [(Sechidis, 2011)][1]
3. The evaluation metrics were implemented in [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) by Szymański et al.

Without their work, this page would not exist.     

[1]: https://doi.org/10.1007/978-3-642-23808-6_10
[2]: https://arxiv.org/abs/1704.08756 
[3]: http://www.uco.es/kdis/mllresources/