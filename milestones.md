

Current it implements: assisted few-shot learning (i.e. where the provided examples are top-k matching the query image with the embedding model)

Now, i want to implement: Hierarchical few-shot learning (i.e. the same as above but first prediction is done with respect to kingdom,phylum,class,order,family,genus and then finally species). So the examples which are provided are at each level of the hierarchy are with respeect to the same level. Please note that each image has its associated kingdom,phylum,class,order,family,genus and species. So one image could potentially have examples at multiple levels of the hierarchy.


