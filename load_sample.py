import numpy


array = numpy.load("./.cache/datasets/the-stack-dedup-v1.2/hugging-face-pretokenized/shard-000.npy")

print("Shape:", array.shape)
print("Size:", array.nbytes / 1024 / 1024, "MB")
print("Data type:", array.dtype)
print("First 10 tokens:", array[:10])
