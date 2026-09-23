# LangChain Runnables

This repository contains my practice and learning work on **Runnables in LangChain**.

The goal of this section is to understand how LangChain Runnables work, how different Runnable primitives can be used, and how they can be combined to build simple LLM workflows and pipelines.

## Topics Covered

### Runnable Primitives

* **RunnableLambda** – Convert a Python function into a Runnable.
* **RunnablePassthrough** – Pass input through a pipeline without modification.
* **RunnableParallel** – Execute multiple Runnables in parallel.
* **RunnableSequence** – Execute multiple Runnables sequentially.
* **RunnableBranch** – Select and execute a Runnable based on a condition.

### Runnable Methods

* `invoke()` – Execute a Runnable with a single input.
* `batch()` – Execute a Runnable with multiple inputs.
* `stream()` – Stream the output as it is generated.

### Runnable Composition

I also practiced combining Runnables using the **pipe (`|`) operator** to create sequential workflows.

Example:

```python
chain = runnable_1 | runnable_2 | runnable_3
```

This allows the output of one Runnable to become the input of the next Runnable.

## Project Structure

```text
Runnables/
│
├── RunnableLambda.py
├── RunnablePassthrough.py
├── RunnableParallel.py
├── RunnableSequence.py
├── RunnableBranch.py
├── LCEL.py
└── README.md
```



## Technologies Used

* Python
* LangChain
* LangChain Core
* LLM APIs
* VS Code
* Git & GitHub

## What I Learned

Through these examples, I learned:

* The basic concept of **Runnables in LangChain**
* How different Runnable primitives work
* How to create Runnables from Python functions
* How to pass data through a chain
* How to run multiple operations in parallel
* How to create sequential workflows
* How conditional branching works
* How to compose multiple Runnables using the `|` operator

## Purpose of This Repository

This repository is part of my **LangChain learning journey**.

I am building my understanding step-by-step by implementing concepts with practical Python examples rather than only studying the theory.

More LangChain concepts and projects will be added as I continue learning.

## Author

**Ayush Singh Rawal**

Learning and building with Python, AI/ML, LangChain, and LLM applications.
