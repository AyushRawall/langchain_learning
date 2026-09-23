from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation"
)

model1=ChatHuggingFace(llm=llm)

model2=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="generate short and simple notes on the following text \n {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="give 5 question and anser on the following text \n {text}",
    input_variables=["text"]

)

prompt3 =PromptTemplate(
    template="merge the provided notes and quiz in a single document \n notes->{notes} and quiz -> {quiz}",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    "notes":prompt1|model1|parser,
    "quiz":prompt2 |model2 |parser
})

merge_chain=prompt3 |model1 |parser

chain=parallel_chain|merge_chain

text="""class sklearn.ensemble.BaggingClassifier(estimator=None, n_estimators=10, *, max_samples=None, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)[source]
A Bagging classifier.

A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [4].

estimatorobject, default=None
The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a DecisionTreeClassifier.


n_estimatorsint, default=10
The number of base estimators in the ensemble.

max_samplesint or float, default=None
The number of samples to draw from X to train each base estimator (with replacement by default, see bootstrap for more details).

If None, then draw X.shape[0] samples irrespective of sample_weight.

If int, then draw max_samples samples.

If float, then draw max_samples * X.shape[0] unweighted samples or max_samples * sample_weight.sum() weighted samples.

max_featuresint or float, default=1.0
The number of features to draw from X to train each base estimator ( without replacement by default, see bootstrap_features for more details).

If int, then draw max_features features.

If float, then draw max(1, int(max_features * n_features_in_)) features.

bootstrapbool, default=True
Whether samples are drawn with replacement. If False, sampling without replacement is performed. If fitting with sample_weight, it is strongly recommended to choose True, as only drawing with replacement will ensure the expected frequency semantics of sample_weight.

bootstrap_featuresbool, default=False
Whether features are drawn with replacement.

oob_scorebool, default=False
Whether to use out-of-bag samples to estimate the generalization error. Only available if bootstrap=True.

warm_startbool, default=False
When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble. See the Glossary.


n_jobsint, default=None
The number of jobs to run in parallel for both fit and predict. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

random_stateint, RandomState instance or None, default=None
Controls the random resampling of the original dataset (sample wise and feature wise). If the base estimator accepts a random_state attribute, a different seed is generated for each instance in the ensemble. Pass an int for reproducible output across multiple function calls. See Glossary.

verboseint, default=0
Controls the verbosity when fitting and predicting.
"""

result=chain.invoke({"text":text})

print(result)

chain.get_graph().print_ascii()