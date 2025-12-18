# %%
from uni_tokenizer import BpeTrainer, PreTokenizer
pre = PreTokenizer(["<|endoftext|>"])
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
from uni_tokenizer import BpeEncoder
import numpy as np
encoder = BpeEncoder.load("test", ch="char")
a = encoder.encode_string("Hello, world!")
print(a)
assert isinstance(a, np.ndarray)
assert a.tolist() == [73, 102, 293, 112, 45, 261, 304, 341, 34]
b = encoder.encode_file("fixtures/tinystories_sample_5M.txt", 100)
print(len(b))
assert len(b) == 2184799
s = encoder.decode(b)
# print(s[:100])
with open("fixtures/tinystories_sample_5M.txt", "rb") as f:
  original = f.read().decode("utf-8")
assert s == original

# %%
encoder = BpeEncoder.load("test", ch="char", special_tokens=[])
a1 = encoder.encode_string("<|endoftext|>")
print(a1)
assert a1.tolist() == [61, 125, 355, 112, 103, 117, 102, 121, 117, 125, 63]

# %%
