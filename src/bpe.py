# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json

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
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

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

        # -- merge -- 
        while(len(self.id_to_token) < self.vocab_size):
            # --- FIXME: 헬퍼 함수로 리팩토링 하기 ---
            # 짝지어진 token pair와 그 횟수를 저장할 딕셔너리 
            pair_dict = {}
            
            # 가장 자주 등장하는 이웃 token pair 찾기
            # 현재 시퀀스 안에서 이웃을 찾는 거기 때문에 vocab_size가 아니라 len(sequence) - 1 까지 순회 
            # len(sequence)까지 순회하면 curr이 마지막 token_id일 때 index + 1 때문에 IndexError 발생 
            for index in range(0, len(sequence) - 1):
                curr_token_id = sequence[index]
                next_token_id = sequence[index + 1]

                # pair 딕셔너리에 key로 사용할 token pair를 저장하는 튜플 
                pair = (curr_token_id, next_token_id)

                # pair를 key로, 등장 횟수를 value로 저장 
                # pair_dict.get(pair, 0) + 1
                # -> pair_dict에 pair가 없으면 0으로 시작 
                # -> pair_dict에 pair가 있으면 기존 횟수 + 1
                pair_dict[pair] = pair_dict.get(pair, 0) + 1

            # max 연산에서 터지지 않기 위한 조건 
            if len(pair_dict) == 0:
                break

            # pair_dict에서 pair_dict.get값 중 가장 큰 get 값을 가진 pair를 찾음 
            best_pair = max(pair_dict, key = pair_dict.get)

            # best_pair의 두 token을 하나의 token으로 합쳐 새로운 token의 token ID 생성
            # 아래 tuple unpacking을 사용하는 게 더 좋은 코드 => 변수 이름을 통해 로직을 직관적으로 알 수 있음 
            # left_token = self.id_to_token[best_pair[0]]
            # right_token = self.id_to_token[best_pair[1]]

            # tuple unpacking 사용 
            left_token_id, right_token_id = best_pair

            # pair의 두 token_id로 token을 얻음
            left_token = self.id_to_token[left_token_id]
            right_token = self.id_to_token[right_token_id]

            # 두 token을 합 해서 새로운 token을 생성
            new_token = left_token + right_token

            # 새로운 token의 token_id 생성 
            # 새로운 token_id 생성 = vocab에서 아직 쓰이지 않은 다음 번호를 부여 -> 260 이후 번호 
            # vocab = id_to_token, token_to_id
            # len(self.id_to_token): 0부터 현재 등록된 id까지의 개수 => 아직 쓰이지 않은 첫 번호의 수를 반환
            new_token_id = len(self.id_to_token)

            # 나중에 현재 sequence로 한 번에 치환할 새로운 리스트 
            new_sequence = []

            # 현재 sequence의 best_pair를 모두 새 token_id로 치환 
            # 원본 sequence 자체를 수정하기보다, 새 리스트를 만들어 저장 후 한꺼번에 치환하는 게 안전
            i = 0

            while(i < len(sequence)):
                if (i + 1) >= len(sequence):
                    new_sequence.append(sequence[i])
                    break
                
                curr_token_id = sequence[i]
                next_token_id = sequence[i + 1]

                # 현재 token이 beat_pair 순서대로 연결되어 있다면  
                if curr_token_id == left_token_id and next_token_id == right_token_id:
                    # 새로운 시퀀스에 new_token_id를 넣어줌 
                    new_sequence.append(new_token_id)

                    # 현재 인덱스와 다음 인덱스를 확인 후 저장했으니 인덱스를 2씩 증가해서 확인하지 않은 인덱스로 이동 
                    i += 2
                # token이 best_pair 순서대로 연결되어 있지 않다면
                else:
                    # 현재 시퀀스 token_id 그대로 새로운 시퀀스에 넣어줌 
                    new_sequence.append(sequence[i])
                
                    # 현재 인덱스를 확인 후 저장했으니 인덱스를 1씩 증가해서 확인하지 않은 인덱스로 이동 
                    i += 1
                
            # 순회가 끝나면 현재 시퀀스를 새로운 시퀀스로 치환 
            sequence = new_sequence
            # --- FIXME: 헬퍼 함수로 리팩토링 하기 ---

            # 새로 만든 token을 vocab에 등록
            # vocab = id_to_token, token_to_id
            # => `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신
            self.id_to_token[new_token_id] = new_token
            self.token_to_id[new_token] = new_token_id
            self.merges.append(best_pair)
        
        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        -> bytes는 정수 리스트로 바꿔서 저장 => list[int]
        -> id_to_token은 key가 int, JSON은 key str => 저장할 때 id를 str로 형 변환 or 리스트 형태로 저장
        -> merges는 tuple -> JSON은 tuple이 없기 때문에 리스트로 저장 

        저장해야 할 핵심 상태
        id_to_token
        token_to_id -> 이건 사실 id_to_token만 있으면 다시 만들 수 있음 
        -> vocab_size를 저장하자 
        merges
        """
        # save 함수에서 Path를 지원하도록 변환
        path = Path(path)

        # 저장할 dict 생성
        data = {}

        # vocab_size 넣기 
        _vocab_size = self.vocab_size
        
        data["vocab_size"] = _vocab_size

        # token 정보를 임시 저장할 딕셔너리 
        token_dict = {}

        # id_to_token을 JSON에 넣을 리스트로 변환해서 넣음
        for item in self.id_to_token.items():
            # int형 id를 str로 형 변환 
            token_dict["_id"] = str(item[0])

            # token이 str일 때
            if isinstance(item[1], str):
                token_dict["_type"] = "str"
                token_dict["_value"] = item[1]

            # token이 bytes일 때 
            elif isinstance(item[1], bytes):
                token_dict["_type"]  = "bytes"
                token_dict["_value"] = list(item[1])

            data["token"] = token_dict

        # merges를 JSON에 넣을 리스트로 변환해서 넣음
        _merges_list = [(left, right) for left, right in self.merges]

        data["merges"] = _merges_list
        
        # path를 열고 json.dump로 저장 
        with open(path, "w") as f:
            json.dump(data, f)
            
        # raise NotImplementedError("BPETokenizer.save를 구현하세요.")

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
        # token id를 저장할 sequence 라는 빈 리스트 만들기
        sequence = []
        
        text_bytes = text.encode("utf-8")
        
        # init에서 만들어둔 기본 byte token ID를 찾아옴 
        for byte_value in text_bytes: 
            token_bytes = bytes([byte_value])

            # 원래 형태로 돌아온 byte로 token id 찾기 
            token_id = self.token_to_id[token_bytes]

            # token id append 해 주기 
            sequence.append(token_id)

        # -- pair를 sequence에 적용 -- 
        # train에서 merges에 저장한 pair를 사용 
        for pair in self.merges:
            # --- FIXME: 헬퍼 함수로 리팩토링 하기 ---
            # 나중에 현재 sequence로 한 번에 치환할 새로운 리스트 
            new_sequence = []
            
            # tuple unpacking 사용 
            left_token_id, right_token_id = pair
        
            # pair의 두 token_id로 token을 얻음
            left_token = self.id_to_token[left_token_id]
            right_token = self.id_to_token[right_token_id]

            # 두 token을 합 해서 merge_token 생성
            merge_token = left_token + right_token

            # merge_token의 token_id 조회 
            merge_token_id = self.token_to_id[merge_token]

            i = 0

            while(i < len(sequence)):
                if (i + 1) >= len(sequence):
                    new_sequence.append(sequence[i])
                    break

                curr_token_id = sequence[i]
                next_token_id = sequence[i + 1]

                # 현재 token이 beat_pair 순서대로 연결되어 있다면  
                if curr_token_id == left_token_id and next_token_id == right_token_id:
                    new_sequence.append(merge_token_id)

                    i += 2
                # token이 best_pair 순서대로 연결되어 있지 않다면
                else:
                    new_sequence.append(sequence[i])
            
                    i += 1

            # 순회가 끝나면 현재 시퀀스를 새로운 시퀀스로 치환 
            sequence = new_sequence
            # --- FIXME: 헬퍼 함수로 리팩토링 하기 ---

        # add_bos_eos가 True면 sequence의 앞에 BOS, 뒤에 EOS 토큰 추가 
        if add_bos_eos:
            # FIXME: get 함수 쓸까
            # 리스트에 정수를 더하려면 반드시 [] 대괄호로 묶어줘야 함 
            sequence = [SPECIAL_IDS[BOS_TOKEN]] + sequence + [SPECIAL_IDS[EOS_TOKEN]]
        
        return sequence
    
        # raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        # 리스트로 만들면 마지막에 decode 메소드 사용 불가
        # -> 처음부터 bytearray 객체로 만들기 
        result_byte = bytearray()

        # 토큰 아이디를 받아서 각 아이디가 가리키는 바이트를 이어 붙이고 마지막에 utf-8 문자열로 복원
        for token_id in ids:
            # 각 id를 통해 id_to_token에서 token 조회
            token = self.id_to_token[token_id]
            
            # skip_special true일 때 
            if skip_special:
                # token이 str이면
                if isinstance(token, str):
                    # 건너뛰기
                    continue
                elif isinstance(token, bytes):
                    result_byte.extend(token)
            # skip_special false일 때 
            else:
                # token이 str이면
                if isinstance(token, str):
                    # token을 utf-8로 encode해서 붙임
                    token_bytes = token.encode("utf-8")
                    result_byte.extend(token_bytes)

                # token이 bytes면
                elif isinstance(token, bytes):
                    result_byte.extend(token)

        # 순회 다 하고 마지막에 decode("utf-8") 호출
        # bytes(result_byte).decode("utf-8") => 불변 bytes 객체로 바꿔서 문자열 디코딩
        # bytes: 수정 불가능, 최종 결과 표현에 좋음 
        # bytearray: 수정 가능, 누적하기에 좋음  
        # -> bytearray도 decode 사용 가능하지만 보통 최종 byte 시퀀스를 만든 뒤 bytes로 확정하고 decode한다는 의미로 이렇게 사용 
        result = bytes(result_byte).decode("utf-8")

        return result
        # raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
