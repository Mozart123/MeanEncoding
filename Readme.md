Mean encoding or impact coding can be useful when a nominal feature has lots of categories.
However, it must be done correctly, otherwise will result in overfitting.

In mean encoding, mean target value corresponding to each category is calculated. However, it is done in such a way that
we don't assign the target means to the examples that we used for calculating these values, while still using all examples for training.
***
How does it work?
Let's say we are not using CV. There is only a single validation set. After that we,
- Split training set into K folds.
- Use in-fold data to calculate the target means for each category.
- Set out-of-fold values to calculated means.
- Add calculated_means/N to test and validation values. (Averaging over N folds.)
- Repeat for all folds.

- If you are using CV, repeat this process for each fold. It will create a K-Fold in K-Fold.

***
Kaggle kernel:
https://www.kaggle.com/ozanozgur/don-t-overfit-to-cv-oversampling-mean-encoding
