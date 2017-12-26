# spark-tools
Some useful tools for Spark ML. Based on Spark 2.2

## BinaryEncoder
Binary encoding for categorical variables, similar to onehot, but stores categories as binary bitstrings.
Somewhat similar to sklearn http://contrib.scikit-learn.org/categorical-encoding/binary.html

Usage is similar to OneHotEncoder and requires StringIndexer to be applied first. Output is a vector representing binary string of category index number. In comparison to OHE the size of the vector is much smaller and is equal to ceil(log2(# of categories))

## IsMissingGenerator
Dataframe transformer used to generate binary columns representing the fact that the original feature is missing (i.e. null or NaN)

## MeanEncoder

