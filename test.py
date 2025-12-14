# %%
from unitoken import BpeTrainer, PreTokenizer
pre = PreTokenizer(["<|endoftext|>"], None)
words = pre.get_words_from_file("fixtures/tinystories_sample_5M.txt", 100)

# %%

bpe = BpeTrainer(["<|endoftext|>"], ch="char")
print(bpe.vocab_size)
bpe.add_words(words)
bpe.train(500)
print(bpe.vocab_size)

# %%
vocabs = dict(bpe.vocabs.items())
vocabs

# %%
bpe.save("test")

# %%
from unitoken import BpeEncoder
import numpy as np
encoder = BpeEncoder(name="test", char_level="char")
a = encoder.encode_string("Hello, world!")
assert isinstance(a, np.ndarray)
assert a.tolist() == [73, 102, 293, 112, 45, 261, 304, 341, 34]
print(len(encoder.encode_file("fixtures/tinystories_sample_5M.txt", 100)))

# %%
