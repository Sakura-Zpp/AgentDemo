import os,hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader


def get_file_SHA256_hex(filepath: str):      #获取文件sha256的64位十六进制字符串
    if not os.path.exists(filepath):
        logger.error(f"文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):
        logger.error(f"路径{filepath}不是文件")
        return

    sha256 = hashlib.sha256()
    chunk_size = 4096
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(chunk_size)
            while chunk:
                sha256.update(chunk)
                chunk = f.read(chunk_size)
            sha256 = sha256.hexdigest()
            return sha256
    except Exception as e:
        logger.error(f"计算文件{filepath}sha256失败,{str(e)}")


def listdir_with_allowed_type(path: str,allowed_types: tuple[str]):        #返回支持的文件列表
    if not os.path.isdir(path):
        logger.error(f"路径 '{path}' 不是有效目录或不存在")
        return tuple()

    files = []
    for f in os.listdir(path):
        full_path = os.path.join(path, f)

        if os.path.isfile(full_path) and f.lower().endswith(allowed_types):
            files.append(os.path.abspath(full_path))

    return tuple(files)


def pdf_load(filepath: str,passwd=None) -> list[Document]:
    return PyPDFLoader(filepath,passwd).load()

def txt_load(filepath: str) -> list[Document]:
    return TextLoader(filepath,encoding="utf-8").load()
