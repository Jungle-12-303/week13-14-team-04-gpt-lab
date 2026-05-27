# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

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
    <pad>: 패딩 토큰
    <unk>: 어휘 사전에 없는 단어 
    <bos>: 텍스트 시작
    <eos>: 텍스트 끝 

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
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        # SPECIAL_IDS에 들어가 있는 특수 토큰들을 id_to_token(encode), token_to_id(decode)로 넣어준다
        for token, token_id in SPECIAL_IDS.items():
            # id_to_token와 token_to_id는 항상 짝!!!! 함께 넣어주기 
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        # bytes([byte_value]) 형태로 등록 -> 순회하면서 byte_value 값이 바뀌어야 한다
        for token_id in range(4, NUM_BYTES + BYTE_OFFSET):
            # 인덱스 값으로 보정 
            byte_value = token_id - BYTE_OFFSET

            # id_to_token와 token_to_id는 항상 짝!!!! 함께 넣어주기 
            self.id_to_token[token_id] = bytes([byte_value])
            self.token_to_id[bytes([byte_value])] = token_id

        # raise NotImplementedError("_init_special_tokens를 구현하세요.")

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
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.

        train = tokenizer 학습 단계
        corpus를 보고 붙어 다니는 byte 쌍을 찾아서, 그 쌍을 새 token으로 등록
        -> corpus: 모델의 훈련을 위해 기본적으로 사용하는 대규모 텍스트 데이터셋 
        => corpus = 원시 텍스트(raw text), 데이터에 특정한 정답(레이블) 정보가 따로 지정되어 있지 않은 일반적인 텍스트 모음

        self.merges = encode가 나중에 같은 규칙으로 병합할 수 있도록 어떤 쌍을 어떤 순서로 합쳤는지 저장하는 목록 

        시퀀스: 현재 corpus를 표현하는 token ID 리스트
        """
        # 기본 token 등록을 위해 _init_special_tokens() 호출 
        self._init_special_tokens()

        # token id를 저장할 sequence 라는 빈 리스트 만들기
        sequence = []

        # byte ID 시퀀스 생성 -> corpus를 utf-8 bytes로 바꿈
        # -> 글자 단위가 아니라 byte 단위로 자르기 위함 = 바이트 페어 인코딩 
        # -> 원본 corpus를 바꾸는 함수가 아니라 새로운 bytes 객체를 반환  
        # 인코딩 된 bytes라 정수 값으로 반환됨 
        corpus_bytes = corpus.encode("utf-8")
        
        # init에서 만들어둔 기본 byte token ID를 찾아옴 
        for byte_value in corpus_bytes: 
            # 인코딩 된 정수 값 bytes를 다시 1-byte짜리 bytes 객체로 감싸서 token_to_id가 조회할 수 있게 해줘야 함
            # bytes([byte_value]) -> byte_value값 하나로 이루어진 bytes 객체를 만듦 
            # => 정수 하나를 byte 값 하나짜리 리스트로 만들고 그 리스트를 bytes 객체로 변환 
            # => 원래 bytes 객체 안에 있던 byte 하나의 형태로 돌아오는 것
            token_bytes = bytes([byte_value])

            # 원래 형태로 돌아온 byte로 token id 찾기 
            token_id = self.token_to_id[token_bytes]

            # token id append 해 주기 
            sequence.append(token_id)

        # 가장 자주 등장하는 이웃 token pair 찾기
        # -> 어디서? 순회를 해야겟지 vocab_size에 도달할 때까지 반복

        # 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환하기
        # 시퀀스가 뭐지 토큰 id를 어떻게 만들지
 
        # `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신
        # 어떤 값을 저기에 넣어야 하는거지 

        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
