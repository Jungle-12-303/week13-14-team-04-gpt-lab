import sys, os
import re

class SimpleTokenizerV1:
  def __init__(self,vocab):
    self.str_to_int = vocab
    self.int_to_str = {i:s for s,i in vocab.items()}
  def encode(self,text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
    preprocessed = [txt.strip() for txt in preprocessed if txt.strip()]
    return_ids = [self.str_to_int[txt] for txt in preprocessed]
    return return_ids
  def decode(self,ids):
    return [self.int_to_str[idx] for idx in ids]

print(os.listdir())
with open(os.path.join(os.path.dirname(__file__),"the-verdict.txt"),"r",encoding="utf-8") as f:
  raw_text = f.read()
# print("총 문자 개수:", len(raw_text))
# print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocabsize = len(all_words)
# print(vocabsize)
vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#   print(item)
#   if i >= 50:
#     break

if __name__ == "__main__":
  tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))