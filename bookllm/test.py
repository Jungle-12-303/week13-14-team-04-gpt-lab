corpus = "이 영화는 정말 좋았다. 이 영화는 다시 보고 싶다."

# `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.

tokenizer = {}
corpus.encode("utf-8")
for i in corpus.encode("utf-8"):
  tokenizer[(i,i+1)]= i

import json
a = {1:bytes([55]),2:(1,2)}
test = json.dumps()
print(test)