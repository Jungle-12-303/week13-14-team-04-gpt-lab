# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.

        여기서는 아직 BPE 학습을 하는 게 아니다
        BPE가 시작할 수 있도록 "모든 UTF-8 byte를 표현할 수 있는 기본 사전"을 깔아두는 단계다
        """
        # train/load를 다시 할 수 있으므로 예전 vocab과 merge rule은 먼저 비운다
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        # enumerate를 쓰면 리스트 순서가 그대로 고정 ID가 된다
        # 즉 <pad>, <unk>, <bos>, <eos>가 항상 0, 1, 2, 3이 된다
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        # BYTE_OFFSET은 지금은 4지만, 의미는 "특수 토큰 영역 바로 다음 ID"다
        # 그래서 byte 0은 ID 4, byte 65는 ID 69처럼 등록된다
        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.

        corpus가 바로 BPE가 규칙을 배울 학습 대상이다
        같은 corpus에서 만든 ids를 while 루프 안에서 계속 압축해 가며 merge rule을 쌓는다
        """
        self._init_special_tokens()

        # corpus를 문자 단위가 아니라 UTF-8 byte token id로 바꾼다
        # 예를 들어 "A"는 byte 65이고, 내부 token id는 BYTE_OFFSET + 65 = 69가 된다
        ids = []
        for byte_value in corpus.encode("utf-8"):
            ids.append(BYTE_OFFSET + byte_value)

        # vocab이 목표 크기에 도달하거나, 더 이상 pair를 만들 수 없을 때까지 반복한다
        # 이 while은 새 문장을 받는 루프가 아니라, 같은 ids를 merge로 점점 압축하는 루프다
        while len(self.id_to_token) < self.vocab_size and len(ids) >= 2:
            pair_counts = {}

            # 현재 ids에서 인접한 두 token id를 pair로 묶어 빈도를 센다
            # 예: A B A B라면 AB, BA, AB가 되고 AB가 2번 나온다
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair not in pair_counts:
                    pair_counts[pair] = 0
                pair_counts[pair] += 1

            # 이번 ids 상태에서 가장 많이 나온 pair 하나만 고른다
            # 하나를 합치면 ids 구조가 바뀌기 때문에, 여러 pair를 한 번에 합치지 않고 다시 통계를 낸다
            best_pair = None
            best_count = 0
            for pair in pair_counts:
                if pair_counts[pair] > best_count:
                    best_pair = pair
                    best_count = pair_counts[pair]

            # 한 번만 나온 pair는 합쳐도 반복 패턴을 배운 효과가 거의 없으므로 멈춘다
            if best_count < 2:
                break

            # 새 token id는 현재 vocab 크기다
            # len(self.id_to_token)을 두 번 직접 쓰면 첫 저장 후 길이가 바뀌므로 new_id로 고정해 둔다
            new_id = len(self.id_to_token)

            # merges는 나중에 encode()에서 같은 순서로 merge를 재현하기 위한 rule 목록이다
            # id_to_token/token_to_id는 decode와 encode에서 ID를 양방향으로 찾기 위한 사전이다
            self.merges.append(best_pair)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id

            # ids를 왼쪽부터 훑으며 best_pair와 일치하는 두 token을 new_id 하나로 바꾼다
            # best_pair를 만나면 token 두 개를 소비했으므로 i를 2 증가시킨다
            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(new_id)
                    i = i + 2
                else:
                    merged.append(ids[i])
                    i = i + 1
            ids = merged

    def save(self, path: str | Path):
        """
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        JSON에 넣기 위해 실제 값은 저장 가능한 list/string 형태로 바꿔 둔다
        """
        path = Path(path)

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": [],
            "merges": [],
        }

        # self.merges는 tuple 목록이다. JSON에는 list 형태로 저장하고, load()에서 다시 tuple로 돌린다
        for pair in self.merges:
            data["merges"].append(list(pair))

        # id_to_token에는 str, bytes, tuple이 섞여 있다
        # 나중에 정확히 복원하려면 값뿐 아니라 원래 type도 같이 저장해야 한다
        for idx, token in self.id_to_token.items():
            if isinstance(token, str):
                token_type = "str"
                token_value = token
            elif isinstance(token, bytes):
                token_type = "bytes"
                token_value = list(token)
            else:
                token_type = "tuple"
                token_value = list(token)

            data["id_to_token"].append({
                "id": idx,
                "type": token_type,
                "value": token_value,
            })

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path):
        """save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다."""
        json_data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab_size = json_data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        # 저장할 때 list가 된 pair를 다시 tuple로 돌린다
        # tuple이어야 token_to_id의 key로 쓸 수 있고, encode에서 pair 비교도 안정적으로 된다
        for pair in json_data["merges"]:
            self.merges.append(tuple(pair))

        # 저장된 type 정보를 보고 str/bytes/tuple 중 원래 형태로 복원한다
        for item in json_data["id_to_token"]:
            idx = int(item["id"])
            token_type = item["type"]
            token_value = item["value"]

            if token_type == "bytes":
                token = bytes(token_value)
            elif token_type == "tuple":
                token = tuple(token_value)
            else:
                token = token_value

            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.

        학습 때와 똑같이 byte ID에서 시작하고, self.merges에 저장된 순서대로 merge를 적용한다
        """
        if not self.id_to_token:
            self._init_special_tokens()

        # 먼저 text를 byte token id로 바꾼다. 여기까지는 train()의 시작과 같은 방식이다
        ids = []
        for byte_value in text.encode("utf-8"):
            ids.append(BYTE_OFFSET + byte_value)

        # BPE는 merge 순서가 중요하다
        # train()에서 배운 순서 그대로 적용해야 같은 텍스트가 같은 token id 배열이 된다
        for pair in self.merges:
            merge_id = self.token_to_id.get(pair)
            if merge_id is None:
                continue

            # 현재 pair가 보이는 자리만 merge_id 하나로 바꾼다
            # train()의 병합 루프와 같은 구조지만, 새 token을 만들지는 않는다
            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    merged.append(merge_id)
                    i = i + 2
                else:
                    merged.append(ids[i])
                    i = i + 1
            ids = merged

        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.

        merge token은 원본 byte token까지 재귀적으로 풀고,
        byte를 하나씩 문자열로 decode하지 말고 마지막에 한 번만 UTF-8 decode한다
        """
        byte_values = []

        for token_id in ids:
            if skip_special and token_id in SPECIAL_IDS.values():
                continue
            byte_values.extend(self._token_to_bytes(token_id))

        # UTF-8 문자는 여러 byte로 이루어질 수 있으므로 byte를 모은 뒤 마지막에 한 번만 decode한다
        return bytes(byte_values).decode("utf-8", errors="replace")

    def _token_to_bytes(self, token_id: int) -> list[int]:
        """token_id 하나를 원본 byte 값 리스트로 풀어낸다"""
        token = self.id_to_token.get(token_id)

        if isinstance(token, bytes):
            return list(token)

        # merge token은 다른 token id 두 개를 묶은 tuple이다
        # 그 안에 또 merge token이 있을 수 있으니 원본 byte가 나올 때까지 재귀적으로 푼다
        if isinstance(token, tuple):
            out = []
            for child_id in token:
                out.extend(self._token_to_bytes(child_id))
            return out

        return []
