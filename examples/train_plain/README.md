# Example: Training a small model on a small dataset

This is a training example with a medium amount of bells and whistles: We have a fairly fully featured training loop, with logging and evaluation, but we make a slightly idiosyncratic (but fast on GPU) choice: Every sample gets padded to a fixed size and then we do all the shuffling on the GPU inside a big `jit`. This is not a good way to work with big datasets, but it's fast for small datasets. For anything more serious, where things don't fit into VRAM, you should consider the `train_grain` example.

**Note: This example runs EXTREMELY slowly on CPU, in particular on Macs. Run it on a GPU if you want it to be fast (or work at all, to be honest). On a H100, this example takes a around 7s in total for training.**
