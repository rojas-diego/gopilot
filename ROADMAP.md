# Roadmap

### 1. Experiment (1 day)

- [ ] Read up on Language Modeling and the related techniques
- [ ] Implement a basic RNN/Transformer on a small Language Modeling dataset such as WikiText2

### 2. Data Loading (1 day)

- [ ] Download and preprocess the stack dataset.
  - [ ] Filter by stars, language, origin
  - [ ] Shrink the size of the dataset
  - [ ] Split final version into chunks and save it on the cloud
  
### 3. Implement and Test the Model (0.5 day)

- [ ] Implement the basic Transformer based LLM
  - [ ] Decide upon an architecture (BERT, GPT, etc)
  - [ ] Pick the number of layers, attention heads, etc
  - [ ] Pick the set of hyperparameters (cf Cramming Paper)
- [ ] Train the model on a subset of our dataset

### 4. Optimize (Optional) (1 day)

- [ ] Use the Cramming paper and related works to come up with optimizations
- [ ] Train a BPE tokenizer on Go data
- [ ] Think of Go specific stuff that could enhance performance
- [ ] Add DeepSpeed support for accelerated training

### 5. Train (1 day)

- [ ] Prepare by setting up checkpointing and monitoring of the run
- [ ] Setup the Docker image to run our training
- [ ] Spin up a VM on the cloud and train for a day
  - [ ] Decide which GPU to use (NVIDIA A40 48 GB, NVIDIA RTX A6000, Smaller?)
  - [ ] Decide on the volume size, number of CPUs, etc

### 6. Fine Tune (Optional) (1-2 days)

- [ ] ELT the APPs dataset and fine-tune our model on coding problems
- [ ] May require a bit more GPU time

### 7. Evaluate (Optional) (1 day)

- [ ] Code a small demo to showcase our project (a small text editor or a vscode extension)
- [ ] Measure its performance on HumanEvalX
