# Roadmap

The goal is to train a "small LLM" on Go code and come up with clever optimisations during training to boost performance. We're trying to investigate how we can make smaller models better in an effort to reverse today's trend of scaling models to trillions of parameters.

### 1. Experiment (1 day)

Get familiar with the task of language modeling and implement a super simple Transformer to setup the repo and to provide us with a safety net before delving into harder stuff.

- [x] Read up on Language Modeling and the related techniques
- [x] Implement a basic RNN/Transformer on a small Language Modeling dataset such as WikiText2 to do a proof of concept.

### 2. Data Loading (1.5 day for it to be really neat)

We want to train our model on Go code. We think the "Stack" dataset might be a good pick as it contains a 30GB+ of Go code from framous repositorites and has been used by other LLM projects. We have to nail this data processing step as there are many challenges with dealing with a big dataset and also many challenges in making sure we're training our model on high quality data. We must come up with a simple tokenizer that we can later improve but we must make sure it contain tokens such as `[SEP]`, `[PAD]`, `[CLS]`, `[MASK]`, `[END]` depending on what we want the final model to be able to do with the code generation ability. For example, a model that has a `[END]` token will know when to stop generating while a model that doesn't might need some other mechanism to make it stop generating stuff. Another example is `[SEP]` which would allow us to pack multiple small sequences in a single sequence fed to the model, the model will learn that this token means "Okay we're switching into a completely different sequence now". This is better than just padding all sequences to a certain length. I alredy did this in part in the `wikitext` example in the repo.

- [x] Download and preprocess the stack dataset (a dataset of code files that has a subset on Go).
  - [x] Filter by stars, language, origin
  - [x] Define a processing pipeline for each sample (whatever is deemed necessary such as dropping certain files because they're redundant, too long, computer generated, etc)
  - [x] Split final version into chunks (files of ~100-500MB) and save it to the cloud
  - [x] Push any preprocessing scripts used in the `./scripts` directory in the Github repository so others can reproduce data preprocessing steps
- [ ] Define a PyTorch `IterableDataset` class that is able to load the postprocessed dataset chunks, this class will be used in the final training step
  - [ ] Define a basic processing pipeline that includes tokenization, transforming the raw file samples into torch padded and batched tensors
  - [ ] Given a list of dataset chunk files, the class iteratively reads from disk and generates batches "on the fly" (in order not to keep 30GB of data in memory)

### 3. Implement and Test the Model (1.5 day)

This step is where we decide on what our "LLM" will be like. We must pick the objective and a bunchhhh of parameters and techniques. Doing a bit of research will definitely help at this stage.

- [ ] Implement the Transformer based LLM we will use for the project.
  - [ ] Decide upon an architecture (BERT, GPT, etc)
  - [ ] Pick the number of layers, attention heads, etc
  - [ ] Pick the set of hyperparameters (cf Cramming Paper, learning rates, scheduling, warmup, additional post processing, and so many other things)
- [ ] Checkout the NanoGPT repository and video for how to implement a LM from scratch if in need of inspiration on how to architecture or code stuff
- [ ] Train the model on our local computers on a subset of our dataset to validate that it works and that it somewhat converges

### 4. Optimize (Optional) (1 day)

In this step, we try to cram in as many optimizations as we can to hopefully improve our yields from the training. This includes optimisations that make the model converge better and optimisations that make training faster. A bunch of optimisations are listed in the Cramming paper and the literature around LLMs is also filled with a bunch of tips and techniques. Our aim is to filter through all that and decide which are worth implementing for us. We can look at a really broad spectrum of optimisations that go from model specific stuff such as parameters to revising the preprocessing logic, improving batching, graddient accumulation, tinkering with the tokenizer and much more. For example, if we train a learned tokenizer on Go data, it will be easier for the model to learn relevant relationship between tokens while keeping a low vocab size. Hopefully, this will increasing downstream performance and accuracy.

- [ ] Use the Cramming paper and related works to come up with optimizations
- [ ] Train a BPE tokenizer on Go data
- [ ] Think of other stuff that could enhance performance
- [ ] Add DeepSpeed support for accelerated training (Microsoft library which enables novel optimizations on NVIDIA GPUs to make Transformers really fast)

### 5. Train (1 day)

Create a container image with all training dependencies so we can just spin up a VM whenever and not get lost in buggy configuration on a remote host. This also means we'll be able to train *anywhere* with a single command allowing us to use any cloud platform such as huawei or other stuff. We must also set up some utilities such as checkpointing, logging, tracking and monitoring of the run so that all training metrics and model weights are available in real-time and safely stored on the cloud for us to access (a bunch of that has already been implemented).

- [ ] Define the Docker image to run our training (it must contain all dependencies and trigger our training script which downloads the dataset and starts feeding batches)
- [ ] Prepare by setting up checkpointing, logging, tracking and monitoring of the run
- [ ] Spin up a VM on the cloud and train for a day
  - [ ] Decide which GPU to use (NVIDIA A40 48 GB, NVIDIA RTX A6000, Smaller?)
  - [ ] Decide on the volume size, number of CPUs, etc
  - [ ] Check on the run every now and then to make sure it hasn't crashed, is converging
  - [ ] Ensure that the GPU is being fully utilized both in terms of memory (if not, we can increase the batch size) and compute (maybe the CPUs are bottlenecking)

### 6. Fine Tune (Optional) (1-2 days)

In this step, we want to optionally fine tune our model on "high quality" data such as asking him to complete a half-implemented function or complete some algorithm. I think the goal of this is to make it learn some algo skills and to refine what it learned. I'm not familiar with fine-tuning tbh. We can ask ChatGPT to come up with a small dataset for fine-tuning our model, Jayden has some cool ideas on that.

- [ ] Fine-tune our model on a handcrafted smaller dataset.
  - [ ] Optionally ask ChatGPT to help us craft this dataset.

### 7. Evaluate (Optional) (1 day)

Right now our model is trained, and I think we'll be able to run it alright on our own computers but a nicer interface could be great to showcase how it performs on day to day tasks. Of course developping a VSCode extension would be ideal and like super nice but it's overkill and might take some time so I think a terminal based thing that just showcases how it autocompletes code snippet is fine. We can also report its performance on Code Generation benchmarks but we must expect it to perform poorly. Finally, we can fine-tune some OpenAI models as per Jayden's suggestion to further study Code LLMs.

- [ ] Code a small demo to showcase our project (a small text editor or a vscode extension)
- [ ] Measure its performance on HumanEvalX
- [ ] Fine tune some models using the OpenAI API
