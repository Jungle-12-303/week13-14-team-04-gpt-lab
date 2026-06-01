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
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self.next_index = BYTE_OFFSET + NUM_BYTES

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        완료
        """
        self.merges = []
        self.id_to_token = {}
        self.token_to_id = {}
        self.next_index = BYTE_OFFSET + NUM_BYTES
        self.id_to_token.update({idx:token for idx,token in enumerate(SPECIAL_TOKENS)})
        for byte_value in range(NUM_BYTES):
            self.id_to_token[BYTE_OFFSET + byte_value] = bytes([byte_value])

        self.token_to_id = {token:idx for idx, token in self.id_to_token.items()}
        
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
        - 가장 자주 등장하는 이웃 token pair를 찾습니다. 완료
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()
        encode_data = list(corpus.encode("utf-8"))
        # encode_data는 단순히 byte를 변환한 데이터만 있어서 255 이하의 한글자 바이트 값 처리가 안됨
        # 그래서 token에 실제 id값이 들어가는데 한글자 이상은 encode_data와 동일(+BYTE_OFFSET 더한 값)
        # 현재 방식은 n개 순환 방식, index 위치를 저장해서 순환하는 방식을 채택할 예정인데
        # 메모리 용량에 따라서 바꿀 필요가 있는지?
        tokens = [idx + BYTE_OFFSET for idx in encode_data]
        # 순환문 시작
        while len(self.id_to_token) < self.vocab_size:
            max_status = self._get_pair_dict(tokens,min_value = 1)
            if max_status:
                self.merges.append(max_status[1])
                self.id_to_token[self.next_index] = max_status[1]
                self.token_to_id[max_status[1]] = self.next_index
                tokens = self._merge_pair(tokens,max_status[1],self.next_index)
                self.next_index += 1
            else:
                break
        
        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def _merge_pair(self,tokens,pair,new_id):
        idx = 0
        result = []
        while idx < len(tokens):
            if (idx < len(tokens) - 1 and (tokens[idx],tokens[idx+1]) == pair):
                result.append(new_id)
                idx += 2
            else:
                result.append(tokens[idx])
                idx += 1
        return result

    def _get_pair_dict(self,tokens,min_value=0):
        """데이터를 받아서 가장 많은 개수의 pair를 찾아서 거기에 대한 정보를 전달

        Args:
            tokens (list): encode_data에 1글자는 byte_offset을 더하고 나머지는 그대로.
            min_value (int, optional): 보류. Defaults to 0.

        Returns:
            max_status or None: 
            pair 개수 <= min_value
            None
            pair 개수 > min_value이고 pair가 제일 많고 먼저 있는 순서
            [pair개수, pair, tokens idx 리스트] 리턴
        """
        pair_dict = {}
        max_value = 0
        max_list = []
        for idx in range(1,len(tokens)):
            pair = (tokens[idx-1], tokens[idx])
            if pair not in pair_dict:
                pair_dict[pair] = [0,[]]
            pair_dict[pair][0] += 1
            pair_dict[pair][1].append(idx-1)
            if max_value < pair_dict[pair][0]:
                max_value = pair_dict[pair][0]
                max_list = [pair]
            elif max_value == pair_dict[pair][0] and pair not in max_list:
                max_list.append(pair)
        if max_value <= min_value:
            return None
        max_status = [-1,None,None]
        for max_pair in max_list:
            if max_status[0] == -1:
                max_status[0] = pair_dict[max_pair][0]
                max_status[1] = max_pair
                max_status[2] = pair_dict[max_pair][1]
            elif pair_dict[max_pair][1][0] < max_status[2][0]:
                max_status[0] = pair_dict[max_pair][0]
                max_status[1] = max_pair
                max_status[2] = pair_dict[max_pair][1]
        return max_status
    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        save_idx = []
        save_token = []
        for idx,token in enumerate(list(self.token_to_id)):
            save_idx.append(idx)
            if idx >= BYTE_OFFSET and idx < NUM_BYTES + BYTE_OFFSET:
                save_token.append(int.from_bytes(token))
            else:
                save_token.append(token)
        json_data = {
            "id" : save_idx,
            "token" : save_token,
            # "convert_id_token_type_data":"",
            "merge_rule" : self.merges,
            # "type_merge_rule" : ""
        }
        json_to_dump = json.dumps(json_data)
        if type(path) == Path:
            path.write_text(json_to_dump,"UTF-8")
        else:
            Path(path).write_text(json_to_dump,"UTF-8")

        # raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        if type(path) == Path:
            data = path.read_text("UTF-8")
        else:
            data = Path(path).read_text("UTF-8")
        raw_json_data = json.loads(data)
        for raw_idx in raw_json_data["id"]:
            idx = int(raw_idx)
            if idx < BYTE_OFFSET:
                data = raw_json_data["token"][idx]
                self.id_to_token[idx] = data
                self.token_to_id[data] = idx
            elif idx < NUM_BYTES + BYTE_OFFSET:
                data = bytes([raw_json_data["token"][idx]])
                self.id_to_token[idx] = data
                self.token_to_id[data] = idx
            else:
                data = tuple(raw_json_data["token"][idx])
                self.id_to_token[idx] = data
                self.token_to_id[data] = idx
        self.merges = list(map(lambda x:tuple(x),raw_json_data["merge_rule"]))
        # raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        encode_data = text.encode("utf-8")
        result_token = [idx + BYTE_OFFSET for idx in encode_data]
        if add_bos_eos:
            result_token = [self.token_to_id[BOS_TOKEN]] + result_token + [self.token_to_id[EOS_TOKEN]]
        for pair in self.merges:
            result_token = self._merge_pair(result_token,pair,self.token_to_id[pair])
        return result_token

        # raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        result_word = b''
        for idx in ids:
            if idx < BYTE_OFFSET:
                if skip_special:
                    pass
                else:
                    result_word = result_word + self.id_to_token[idx].encode("utf-8")

            elif idx >= BYTE_OFFSET + NUM_BYTES:
                result_word = result_word + self.decode_token(idx, skip_special)
            else:
                result_word = result_word + self.id_to_token[idx]
        return result_word.decode("utf-8")
        # raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
        
    def decode_token(self, token_id: int, skip_special:bool) -> bytes:
        token = self.id_to_token[token_id]
        
        if isinstance(token, bytes):
            return token

        if isinstance(token, tuple):
            left, right = token
            return self.decode_token(left,skip_special) + self.decode_token(right,skip_special)

        if isinstance(token, str):
            if skip_special:
                return b""
            return token.encode("utf-8")

        raise TypeError(f"Unknown token type: {type(token)}")