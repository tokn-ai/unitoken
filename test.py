# %%
from unitoken import BpeTrainer, PreTokenizer
pre = PreTokenizer(["<|endoftext|>"], None)
words = pre.get_words_from_file("fixtures/tinystories_sample_5M.txt", 100)

# %%

bpe = BpeTrainer(["<|endoftext|>"], ch="char")
assert bpe.vocab_size == 257
bpe.add_words(words)
bpe.train(500)
assert bpe.vocab_size == 500

# %%
vocabs = dict(bpe.vocabs.items())
assert len(vocabs) == 500

# %%
bpe.save("test")

# %%
from unitoken import BpeEncoder
import numpy as np
encoder = BpeEncoder.load("test", char_level="char")
a = encoder.encode_string("Hello, world!")
print(a)
assert isinstance(a, np.ndarray)
assert a.tolist() == [73, 102, 293, 112, 45, 261, 304, 341, 34]
b = encoder.encode_file("fixtures/tinystories_sample_5M.txt", 100)
print(len(b))
assert len(b) == 2249369

# %%
