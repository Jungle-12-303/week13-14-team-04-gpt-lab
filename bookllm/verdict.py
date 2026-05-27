import sys, os
import re
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
for i, item in enumerate(vocab.items()):
  print(item)
  if i >= 50:
    break

