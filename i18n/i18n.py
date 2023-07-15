import locale
import json
import os
from loguru import logger

'''
本地化方式如下所示
LANGUAGE_AND_MODEL_PATH中添加不同模块的本地化信息
'''

LANGUAGE_AND_MODEL_PATH = {
    "gui_realtime.py": {  # 此模块是gui.py的语言文件
        "path": "i18n/gui_realtime",  # 此目录下以语言名的json文件储存本地化信息
        "base_language": "zh_CN",  # base语言会和代码同步更新
        "language_list": ['zh_CN', 'en_US']  # 支持的语言需要在这里记录
    }
}


def read_json_to_map(path: str):
    _file = open(path, 'r', encoding="UTF-8")
    _map: dict = json.load(_file)
    return _map


class I18nAuto:
    def __init__(self, model: str, language: str = None):
        # 读写死的配置,并提取所需信息
        self.LANGUAGE_AND_MODEL_PATH = LANGUAGE_AND_MODEL_PATH
        assert model in self.LANGUAGE_AND_MODEL_PATH.keys()
        self.model_info = self.LANGUAGE_AND_MODEL_PATH[model]
        # 如果language是None则根据本机自动选择语音, 如果是不支持的语言则自动选择base
        if language is None:
            language = locale.getdefaultlocale()[0]
        if language not in self.model_info["language_list"]:
            language = self.model_info["base_language"]
        self.language = language
        logger.info(f"Loading language {self.language} for model {model}.")
        # 首先读base语言,并断言其继承值为"SUPER"标识其为base语言
        base_path = os.path.join(self.model_info["path"], self.model_info["base_language"] + ".json")
        self.map = read_json_to_map(base_path)
        assert self.map["SUPER"] == "SUPER"
        # 然后用pre_lang_map_from_path方法读所使用的语言
        language_path = os.path.join(self.model_info["path"], language + ".json")
        self.map.update(self.pre_lang_map_from_path(language_path))

    def pre_lang_map_from_path(self, lang_map_path: str):
        output_map = {}
        input_map = read_json_to_map(lang_map_path)
        super_language = input_map["SUPER"]
        if super_language in ("SUPER", "END"):  # 如果目标语言不继承别的语言或是base语言,则直接返回
            output_map = input_map
            return output_map
        else:  # 如果目标语言继承自别的语言,则递归读取,并断言其父语言是存在的
            assert super_language in self.model_info["language_list"]
            super_map_path = os.path.join(self.model_info["path"], super_language + ".json")
            output_map.update(self.pre_lang_map_from_path(super_map_path))
            output_map.update(input_map)
            return output_map

    def __call__(self, raw_str):
        return self.map[raw_str]
