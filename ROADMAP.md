# Roadmap

The goal is to train a "micro LLM" on Go code and come up with clever optimisations to boost performance. We're trying to investigate how we can make smaller models better in an effort to reverse today's trend of scaling models to trillions of parameters.

### 1. Experiment (1 day)

Get familiar with the task of language modeling and implement a super simple Transformer to setup the repo and to provide us with a safety net before delving into harder stuff.

- [x] Read up on Language Modeling and the related techniques
- [x] Implement a basic RNN/Transformer on a small Language Modeling dataset such as WikiText2 to do a proof of concept.

### 2. Data Loading (1.5 day for it to be really neat)

In this stage, we must load and process our chosen dataset (The Stack Dedup) and settle upon a Tokenizer that is able to represent code efficiently.

- [x] Download and preprocess the stack dataset (a dataset of code files that has a subset on Go).
  - [x] Filter by stars, language, origin
  - [x] Define a processing pipeline for each sample (whatever is deemed necessary such as dropping certain files because they're redundant, too long, computer generated, etc)
  - [x] Split final version into chunks (files of ~200MB) and save it to the cloud
  - [x] Push any preprocessing scripts used in the `./scripts` directory in the Github repository so others can reproduce data preprocessing steps
- [x] Define a PyTorch `IterableDataset` class that is able to load the postprocessed dataset chunks, this class will be used in the final training step
  - [x] Define a basic processing pipeline that includes tokenization, transforming the raw file samples into torch padded and batched tensors
  - [x] Given a list of dataset chunk files, the class iteratively reads from disk and generates batches "on the fly"
- [x] Define a Go-optimised tokenizer
  - [x] Train a BPE tokenizer on Go code
  - [ ] Think of clever tricks to make the tokenizer as efficient as possible.

### 3. Implement and Test the Model (1.5 day)

This step is where we decide on what our "LLM" will look like. We must define the objective and a tune a bunchhhh of parameters. Doing a bit of research will definitely help at this stage.

- [x] Implement the Transformer based LLM we will use for the project.
  - [x] Decide upon an architecture (BERT, GPT, etc)
  - [x] Pick the number of layers, attention heads, etc
  - [ ] Pick the set of hyperparameters (cf Cramming Paper, learning rates schedues, scheduling, warmup, additional post processing, and so many other things)
- [x] Checkout the NanoGPT repository and video for how to implement a LM from scratch if in need of inspiration on how to architecture or code stuff
- [x] Train the model on our local computers on a subset of our dataset to validate that it works and that it somewhat converges

### 4. Optimize (Optional) (1 day)

This step aims at optimising our model to make sure it processes the most amount of token per second and makes the most of the limited training window. We can leverage layer fusion, optimisation libraries, profilers and a bunch of other techniques.

- [ ] Use the Cramming paper and related works to come up with optimizations
- [ ] Think of other stuff that could enhance performance
- [ ] Add DeepSpeed support for accelerated training (Microsoft library which enables novel optimizations on NVIDIA GPUs to make Transformers really fast)

### 5. Train (1 day)

Prepare the production run. Must rigourously setup tracking, checkpointing, logging, monitoring and validate the compute environment.

- [ ] Define the Docker image to run our training (it must contain all dependencies and trigger our training script which downloads the dataset and starts feeding batches)
- [x] Prepare by setting up checkpointing, logging, tracking and monitoring of the run
- [ ] Spin up a VM on the cloud and train for a day
  - [ ] Decide which GPU to use (NVIDIA A40 48 GB, NVIDIA RTX A6000, Smaller?)
  - [ ] Decide on the volume size, number of CPUs, etc
  - [ ] Check on the run every now and then to make sure it hasn't crashed, is converging
  - [ ] Ensure that the GPU is being fully utilized both in terms of memory (if not, we can increase the batch size) and compute (maybe the CPUs are bottlenecking)

### 6. Fine Tune (Optional) (1-2 days)

Use high quality datasets to fine tune our model on efficient code completion and algorithms.

- [ ] Fine-tune our model on a handcrafted smaller dataset.
  - [ ] Optionally ask ChatGPT to help us craft this dataset.

### 7. Evaluate (Optional) (1 day)

We must define how we decide to evaluate our models. Some benchmarks out there test Go code completion abilities, but there's a risk we'll get 0% accuracy.

- [ ] Code a small demo to showcase our project (show how the model completes every day code)
- [ ] Measure its performance on HumanEvalX (likely to get 0% accuracy)
- [ ] Measure performance on our own handmade evalutation dataset (can ask GPT-3/4)
- [ ] Fine tune some models using the OpenAI API (for comparison)
