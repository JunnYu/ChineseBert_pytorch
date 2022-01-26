# junnyu
from functools import lru_cache

from pypinyin import Style, pinyin
from transformers.models.bert import BertTokenizerFast

import chinesebert.transformers_patch_utils


class ChineseBertTokenizerFast(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinyin_dict = kwargs.get("pinyin_map")
        self.id2pinyin = kwargs.get("id2pinyin")
        self.pinyin2tensor = kwargs.get("pinyin2tensor")
        self.special_tokens_pinyin_ids = [0] * 8

    def custom_input_ids(self, text_or_text_pair, e):
        custom_input_ids = {"pinyin_ids": self.get_pinyin_ids(text_or_text_pair, e)}
        return custom_input_ids

    # pinyin_ids
    @lru_cache(maxsize=2021)
    def get_pinyin_locs_map(self, text=None):
        if text is None:
            return None
        pinyin_list = pinyin(
            text,
            style=Style.TONE3,
            heteronym=True,
            errors=lambda x: [["not chinese"] for _ in x],
        )
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids
        return pinyin_locs

    def get_pinyin_ids(self, text_or_text_pair, e):
        if isinstance(text_or_text_pair, tuple):
            text, text_pair = text_or_text_pair
        else:
            text, text_pair = text_or_text_pair, None

        text_pinyin_locs = self.get_pinyin_locs_map(text)
        text_pair_pinyin_locs = self.get_pinyin_locs_map(text_pair)

        pinyin_ids = []
        special_token_count = 0

        for offset in e.offsets:
            if offset == (0, 0):
                special_token_count += 1
                pinyin_ids.extend([0] * 8)
                continue
            if offset[1] - offset[0] != 1:
                pinyin_ids.extend([0] * 8)
                continue

            if special_token_count <= 1:
                pinyin_locs_maps = text_pinyin_locs
            else:
                pinyin_locs_maps = text_pair_pinyin_locs

            if offset[0] in pinyin_locs_maps:
                pinyin_ids.extend(pinyin_locs_maps[offset[0]])
            else:
                pinyin_ids.extend([0] * 8)

        return pinyin_ids


if __name__ == "__main__":
    tokenizer = ChineseBertTokenizerFast.from_pretrained("junnyu/ChineseBERT-base")
    text1 = [
        "中新网客户端北京9月19日电 (记者 李金磊)中秋三天假期明天开启，国庆七天假期也不远了。“双节”期间，不少小伙伴可能因为工作需要还得加班。那么，中秋国庆假期加班，加班工资该怎么算呢？",
        "多地人社部门已经明确，今年9月21日(中秋节)和10月1日、10月2日、10月3日(国庆节)，这4天是法定节假日，只要安排加班，就应当支付不低于300%工资。",
        "需要注意的是，这4天加班不能用补休来替代加班工资，必须发放3倍工资。",
        "也就是说，这6天假期用人单位安排劳动者加班的，应先安排补休；不能安排补休的，应按不低于工资的200%支付加班工资报酬。",
    ]
    text2 = [
        "如果你中秋国庆假期都在加班，而且没有安排补休，那么，中秋国庆加班10天，可以拿到相当于24天的日工资。",
        "这个中秋国庆假期，你会加班吗？如果你不会加班，选择出行，那么需要这些事情。",
        "交通运输部表示，中秋假期期间，7座以下(含7座)小型客车通行收费公路将正常收费，公众出行以中短途探亲、旅游为主，全国高速公路网流量预计3300万辆左右，与2019年同期基本持平。",
        "文化和旅游部提醒，要密切关注国内疫情动态和中高风险地区变化情况，不要前往中高风险地区旅游。旅游行程中要充分理解并自觉遵守查验健康码、测温、规范戴口罩等防控要求",
    ]

    outputs = tokenizer(
        text1, text2, return_tensors="pt", max_length=256, padding=True, truncation=True
    )
    for k, v in outputs.items():
        if k == "pinyin_ids":
            v = v.reshape(v.shape[0], -1, 8)
        print(k, ":", v.shape)

    # input_ids : torch.Size([4, 141])
    # pinyin_ids : torch.Size([4, 141, 8])
    # token_type_ids : torch.Size([4, 141])
    # attention_mask : torch.Size([4, 141])

    outputs = tokenizer(
        text1,
        text2,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    for k, v in outputs.items():
        if k == "pinyin_ids":
            v = v.reshape(v.shape[0], -1, 8)
        print(k, ":", v.shape)

    # input_ids : torch.Size([4, 256])
    # pinyin_ids : torch.Size([4, 256, 8])
    # token_type_ids : torch.Size([4, 256])
    # attention_mask : torch.Size([4, 256])
