---
layout: post
title: Standardizing Traditional Machine Learning pipelines to Tensor Computation using Hummingbird
subtitle: Application of Hummingbird package from Microsoft Researcher
cover-img: /assets/img/hummingbird.jpg
share-img: /assets/img/hummingbird.jpg
tags: [Machine Learning, Hummingbird, Pytorch, Microsoft]
---
# Background and challenges 📋

There is merely no day passing without Machine learning, Deep learning, and Artificial intelligence mentioned in the news. When talking about Traditional ML, I want to mention the foundation algorithms that have been developed for a long time; in contrast with Deep Neural Networks (DNNs). Despite the rapid growth of Deep learning in recent years at many aspects like Computer Vision, Natural Language Processing, or Recommendation System,... Traditional Machine learning is still completely dominant.
![hb_kaggle](/assets/img/hb_kaggle.png)
Another survey studying the “Most used libraries” within 1.2M publicly available Data science notebooks on GITHUB in July 2017 (GH2017) and 5.1M notebooks in July 2019 (GH2019) indicated that the four most widely used libraries are NumPy, Matplotlib, Pandas, and Scikit-learn (not DNNs frameworks). Moreover, Scikit-learn is roughly 5 times more prominent than PyTorch and TensorFlow combined and is growing faster than both.

The main reason why they are still on the top of attention is the capability of interpretation of the black box for making the right indication for business, simpler, and more efficient usage in software infrastructure. Recently, not only the Big but also Small and Medium-sized enterprises are approaching Machine Learning to help solve business challenges including predictive maintenance, customer churn prediction, and supply-chain optimizations. For business, they prefer understanding the data, studying the characteristics of each parameter and considering them thoroughly rather than solely heading toward highly accurate predictions.

However, there is a critical path of interactive or analytical enterprise applications, which is Model scoring. Among the total cost of a data science solution, model scoring accounts for 45–65% (based on AWS report [1]). Commonly, when deployment, even during training, ML models are scored many times and applied in real work applications (e.g., scale-out batch or interactive serving, personal computers, mobile and IoT devices). Due to portability, maintainability, deployment, and monitoring concerns, scoring dominates the complexity. Hence, latency (timing) and throughput for scoring models are big concerns for enterprises.

Up to now, the most popular toolkits like ML.NET (.NET-based), Scikit-learn (Python-based), and H2O (Java-based)to generate a predictive pipeline of operators such as trained models, preprocessors, featurizers, missing-value imputers, but they are primarily optimized for training, not for scoring on test set during deployment. These frameworks’ portability is typically limited for supporting their operator in many environments. One of the solutions to optimize the Model scoring is to take the advance of Directed Acyclic Graphs (DAGs) from DNNs by transforming the complex and diverse of Traditional ML into a small set of simple tensor. Based on this idea, many systems had been built like ONNX Runtime [2], TorchScript [3], and TVM [4], capitalizing on the relative computational simplicity of neural networks DNN runtime optimizations but they are still limited in capability and applicability to work with Traditional ML.

Recently, Hummingbird - a library from the Microsoft researcher team - was released which can compile featurization operators and Traditional ML models into a small set of tensor operations to enhance the efficient computations for both CPU and hardware accelerators (GPU, TPU). It opens the new way to reduce infrastructure complexity and model scoring cost of Traditional ML. In the next following sections, the highlight of System Architecture and Implementation in terms of three fundamental research questions will make you have clearly an overview of it.

```Can traditional ML operators (both linear algebra-based such as linear models, and algorithmic ones such as decision trees) be translated to tensor computations?```

```Can the resulting (dense) computations in tensor space be competitive with the (sparse) imperative alternatives we get as input (e.g., traversing a tree)?```

```Can HUMMINGBIRD help in reducing software complexity and improving model portability?```

# Technique highlight 📄
![hb_frame](/assets/img/hb_frame.png)
Hummingbird library was based on the simple observation on the Traditional ML pipeline cycle life. Once featurizers is applied and the model is trained, whole of the pipeline can be symbolized as a function transforming input features into a prediction score (e.g., True or False for binary classification). So, the aim of Hummingbird is to compile the prediction functions (not the training algorithms) for each operator in the pipeline into tensor computations and tie them appropriately. To do so, Hummingbird introduces three main components:

1. **Pipeline Parser**: generates an in-memory Intermediate Representation (IR) object encoding each operator in a given predictive pipeline with its input parameters and related input/output dependencies.
2. **Optimizer**: searches potential compilation strategy for an operator and produce a potentially modified IR.
3. **Tensor DAG Compiler:**:  picks the optimized IR object and compiles it into tensor operations following the target DNN runtime format.

Hummingbird currently supports compiling many representative algorithmic operators into tensor computations. In practice, no strategy consistently dominates the others, it depends on each input and model structure. For more detail, you can check out their [Github](https://github.com/microsoft/hummingbird) or this [paper](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf).
In this article, the approach of Hummingbird on Tree-based Models was introduced. There are three different strategies for compiling Tree-based Models: GEneric Matrix Multiplication (GEMM), TreeTraversal, and PerfectTreeTraversal.

* **GEMM** strategy can run highly efficiently with models having small batch size or a large number of small trees (D≤ 3 on CPU, D≤ 10 on GPU - D is the height of the tree).
* **TreeTraversal**: with large batch sizes and taller trees (D > 10), TreeTraversal strategy typically outperforms the GEMM strategy.
* **PerfectTreeTraversal** is slightly faster than TreeTraversal due to the reduced number of index lookups and better coalesced memory accesses but if the trees are too deep, PerfectTreeTraversal get trouble with memory footprint (D≤ 10).

Let’s looking at a simple example of how Hummingbird compile simple regression tree with GEMM strategy work. In this example, the tree takes input as a feature vector with five elements (x∈R5), four decision nodes (orange), and five leaf nodes (blue). Hummingbird will translate the decision tree model into neural networks as below:
![hb_example](/assets/img/hb_example.png)
Assume that we want to calculate the output of this observation
![hb_input](/assets/img/hb_input.png)

* **Step 1**: multiply input tensor with tensor A (computed from the Decision tree model above) that captures the relationship between input features and internal nodes. Then compare it with tensor B which is set to the value of threshold each internal node to create the tenor input path that represents the path from input to node. In this case, the tree model has 4 conditions and the input vector is 5, therefore, the shape of tensor A is 5x4 and tensor B is 1x4.
![hb_step1](/assets/img/hb_step1.png)
* **Step 2**: the tensor p will be multiplied with tensor C that captures whether the internal node is a parent of that internal node, and if so, whether it is in the left or right sub-tree (left = 1, right =-1, otherwise =0) and then check the equals with tensor D that captures the count of the left child of its parent in the path from a leaf node to the tree root to create the tenor output path that represents the path from node to output. In this case, this tree model has 5 outputs with 4 condition, therefore, the shape of tensor C is 4x5 and tensor D is 1x5.
![hb_step2](/assets/img/hb_step2.png)
* **Step 3**: Tensor P will be multiplied with tensor E that captures the mapping between leaf nodes to infer the final prediction. In this case, tree model has 5 outputs, therefore, shape of tensor E is 5x1.
![hb_step3](/assets/img/hb_step3.png)

From this example, we can clearly imagine how Hummingbird compiles a Tree based model. Furthermore, there are other techniques that Hummingbird to efficiently compile Traditional ML like Automatic Broadcasting, Minimize Operator Invocations,… Currently Hummingbird supports over 40 scikit-learn operators
![hb_support](/assets/img/hb_support.png)

Even Hummingbird is very useful and saves ton of time and money for enterprise and data scientist, it is still under development and has a lot of limitations that need more contribution from the community to improve:

* Do not support arbitrary user-defined operators
* Do not support sparse data well
* Do not support text feature extraction

# Experimental evaluation 📊
Let’s test the performance of Hummingbird in regression model with California Housing dataset. Total observation is 20640 with 8 numeric input features and the target variable is the median house value. This dataset was obtained from the StatLib repository, you can also download using the ```sklearn.datasets.fetch_california_housing function```.

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/vumichien/hummingbird/v/3&cellId=1" title="Jovian Viewer" height="800" width="800" frameborder="0" scrolling="auto"></iframe>

In this experiment, 4 different models from simple to complex one was testing, these are LinearRegression, DecisionTreeRegressor, RandomForestRegressor, and XGBRegressor. The metrics to be used are MSE and Run time, which is shown at the end of the post.

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/vumichien/hummingbird/v/3&cellId=2" title="Jovian Viewer" height="800" width="800" frameborder="0" scrolling="auto"></iframe>

It’s no doubt that with the complex model, the accuracy will be improved but the training time, as well as the scoring time also increase. Let’s compile these models to Tensor operator and check whether the compilation will help reduce the running time or not. First, these converted models will run in CPU.

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/vumichien/hummingbird/v/3&cellId=3" title="Jovian Viewer" height="750" width="800" frameborder="0" scrolling="auto"></iframe>

The running time didn’t reduce in all cases. The reason is the number of testing data is not big and the complexity of each model itself is normal. Let’s see the performance of converted models running in GPU.

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/vumichien/hummingbird/v/3&cellId=4" title="Jovian Viewer" height="600" width="800" frameborder="0" scrolling="auto"></iframe>

In this time, the running time was reduced compared to running in CPU. However, comparing with the original model, only the running time of RandomForestRegressor, and XGBRegressor were reduced, the running time of LinearRegression and DecisionTreeRegressor were still higher given the simplicity in the models themselves. The summary of all results is below.

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/vumichien/hummingbird/v/3&cellId=5" title="Jovian Viewer" height="200" width="800" frameborder="0" scrolling="auto"></iframe>

# Final Thoughts 📕
It’s clear to see the trade-off between the complexity of the model (or the accuracy) with the running time. Converting the Traditional model to Tensor operator in some cases doesn’t help to reduce the scoring time due to the number of testing data, the complexity of the dataset, and the complication of the model. You can freely to testing the performance of the compiled model with more complex Featurizers in pipeline model such as using Transform Technique (RobustScaler, StandardScaler, KBinsDiscretizer,..), Imputation Technique (SimpleImputer, MissingIndicator,..) or Feature extraction Technique (PolynomialFeatures, FeatureHasher,..) in your all dataset to see the efficiency of converting to Tensor operator.

You can check more detail in [here](https://towardsdatascience.com/standardizing-traditional-machine-learning-pipelines-to-tensor-computation-using-hummingbird-7a0b3168670)

Enjoy!!! 👦🏻

# References
[1] Amazon. The total cost of ownership (tco) of amazon sagemaker. https://pages.awscloud.com/rs/112-TZM-766/ images/Amazon_SageMaker_TCO_uf.pdf, 2020.  <br/>
[2] ONNX Runtime. https://github.com/microsoft/onnxruntime.  <br/>
[3] TorchScript Documentation. https://pytorch.org/docs/stable/jit.html.  <br/>
[4] T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, M. Cowan, H. Shen, L. Wang, Y. Hu, L. Ceze, C. Guestrin, and A. Krishnamurthy. Tvm: An automated end-to-end optimizing compiler for deep learning, 2018 <br/>


