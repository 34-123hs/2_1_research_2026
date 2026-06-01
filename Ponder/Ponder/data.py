"""data.py — MemmapDataset + TiktokenHFWrapper copied verbatim from AMOE/model.py."""
import tiktoken
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TiktokenHFWrapper(PreTrainedTokenizer):
    """
    tiktoken r50k_base(GPT-2 BPE, ~50k vocab)를 HuggingFace PreTrainedTokenizer
    인터페이스로 감싸 DataCollatorForLanguageModeling 등과 호환되게 한다.
    """

    vocab_files_names = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, encoding_name="r50k_base", **kwargs):
        """
        input : encoding_name (str) tiktoken 인코딩 이름
        output: 없음. eos/bos/unk/pad 토큰을 모두 <|endoftext|>로 설정.
        """
        self._enc = tiktoken.get_encoding(encoding_name)
        self._eot = self._enc.eot_token
        eot_str = "<|endoftext|>"
        kwargs.setdefault("eos_token", eot_str)
        kwargs.setdefault("bos_token", eot_str)
        kwargs.setdefault("unk_token", eot_str)
        kwargs.setdefault("pad_token", eot_str)
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        """output: (int) vocab 크기."""
        return self._enc.n_vocab

    def get_vocab(self):
        """output: dict {decoded_token_str: id} (전체 vocab)."""
        return {self._enc.decode([i]): i for i in range(self.vocab_size)}

    def _tokenize(self, text):
        """input: text (str) → output: list[str] (id를 문자열화한 토큰)."""
        return [str(i) for i in self._enc.encode(text, allowed_special={"<|endoftext|>"})]

    def _convert_token_to_id(self, token):
        """input: token (str) → output: (int) id."""
        return int(token)

    def _convert_id_to_token(self, index):
        """input: index (int) → output: (str) 토큰."""
        return str(index)

    def convert_tokens_to_string(self, tokens):
        """input: tokens (list[str]) → output: (str) 디코딩된 텍스트."""
        return self._enc.decode([int(t) for t in tokens])

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """tiktoken은 별도 vocab 파일이 없으므로 빈 튜플 반환."""
        return ()


class MemmapDataset(Dataset):
    """
    사전 토크나이즈된 uint16 바이너리(train.bin / val.bin)를 numpy.memmap으로 읽어
    block_size 윈도우로 슬라이스해 제공하는 데이터셋.
    """

    def __init__(self, path, block_size, dtype=np.uint16, max_tokens=None):
        """
        input : path (str) .bin 경로, block_size (int) 윈도우 길이,
                dtype 토큰 dtype, max_tokens (int|None) 사용할 토큰 상한
        output: 없음.
        """
        self.data = np.memmap(path, dtype=dtype, mode="r")     # [n_tokens]
        self.block_size = block_size

        n_tokens = len(self.data)
        if max_tokens is not None:
            n_tokens = min(n_tokens, max_tokens)

        self.n_blocks = n_tokens // block_size

    def __len__(self):
        """output: (int) 블록(샘플) 개수."""
        return self.n_blocks

    def __getitem__(self, idx):
        """
        input : idx (int)
        output: dict {"input_ids": LongTensor [block_size]}
        """
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.from_numpy(self.data[start:end].astype(np.int64))   # [block_size]
        return {"input_ids": x}
