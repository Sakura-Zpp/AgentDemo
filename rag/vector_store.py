import os.path
import pickle

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from utils.config import chroma_config
from utils.path_tool import get_abs_path
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import txt_load, pdf_load, listdir_with_allowed_type,get_file_SHA256_hex
from utils.logger_handler import logger


class VectorStore:
    def __init__(self):
        self.vectors = Chroma(
            collection_name=chroma_config['collection_name'],
            embedding_function=embedding_model,
            persist_directory=get_abs_path(chroma_config['persist_directory']),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config['chunk_size'],
            chunk_overlap=chroma_config['chunk_overlap'],
            separators=chroma_config['separators'],
            length_function=len,
        )
        self.bm25_documents: list[Document] = []

        self.bm25_persist_path = get_abs_path(chroma_config['persist_directory']) + "/bm25_index.pkl"

        self._load_bm25_index()

    def _load_bm25_index(self):
        """加载 BM25 文档列表"""
        if os.path.exists(self.bm25_persist_path):
            try:
                with open(self.bm25_persist_path, 'rb') as f:
                    self.bm25_documents = pickle.load(f)
                logger.info(f" 加载 {len(self.bm25_documents)} 个 BM25 文档")
            except Exception as e:
                logger.error(f" 加载 BM25 失败：{e}")
                self.bm25_documents = []
        else:
            logger.info("ℹ 未找到 BM25 文件")

    def _save_bm25_index(self):
        """保存BM25文档"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.bm25_persist_path), exist_ok=True)
            with open(self.bm25_persist_path, 'wb') as f:
                pickle.dump(self.bm25_documents, f)
            logger.info(f" 已保存 {len(self.bm25_documents)} 个 BM25 文档")
        except Exception as e:
            logger.error(f" 保存 BM25 失败：{e}")

    def get_retriever(self):
        vector_retriever = self.vectors.as_retriever(search_kwargs={"k": chroma_config['k']})
        bm25_retriever = BM25Retriever.from_documents(documents=self.bm25_documents,k=chroma_config['k'])
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=chroma_config['weights'],
        )

        return ensemble


    def load_document(self):
       """
       从数据文件读取数据，转为向量存入数据库，
       并对文件进行去重
       :return:None
       """

       def check_sha256_hex(sha256_hex: str):
           if not os.path.exists(get_abs_path(chroma_config['sha256_hex_store'])):
               #创建文件
               open(get_abs_path(chroma_config['sha256_hex_store']), 'w',encoding="utf-8").close()
               return False             #文件不存在返回false

           with open(get_abs_path(chroma_config['sha256_hex_store']), 'r', encoding="utf-8") as f:
               for line in f:
                   if line.strip() == sha256_hex:
                       return True      #文件存在，且哈希值匹配返回ture

               return False             #文件存在，但文件中没有匹配的哈希值

       def save(sha256_hex: str):
           with open(get_abs_path(chroma_config['sha256_hex_store']), 'a', encoding="utf-8") as f:
               f.write(sha256_hex + "\n")

       def get_file_doc(read_path: str):
           if read_path.endswith(".txt"):
               return txt_load(read_path)

           if read_path.endswith(".pdf"):
               return pdf_load(read_path)

           return []

       allowed_file_path:tuple[str] = listdir_with_allowed_type(
           get_abs_path(chroma_config['data_path']),
           tuple(chroma_config['allowed_file_type'])
       )

       for path in allowed_file_path:
           #获取哈希值
           SHA256_hex = get_file_SHA256_hex(path)

           if check_sha256_hex(SHA256_hex):
              logger.info(f"加载知识库{path}已存在,跳过")
              continue

           try:
               documents: list[Document] = get_file_doc(path)

               if not documents:
                   logger.error(f"{path}没有内容，跳过")
                   continue

               split_doc = self.spliter.split_documents(documents)

               if not split_doc:
                   logger.error(f"{path}没有内容，跳过")
                   continue

               #将内容存入数据库
               self.vectors.add_documents(split_doc)
               self.bm25_documents.extend(split_doc)
               self._save_bm25_index()
               #记录已处理文件哈希值，避免重复
               save(SHA256_hex)

               logger.info(f"{path},内容加载成功")
           except Exception as e:
               #exc_info为true会记录详细报错
               logger.error(f"{path}数据库加载失败：{str(e)}",exc_info=True)

if __name__ == '__main__':
    vs = VectorStore()
    vs.load_document()
    retrieve = vs.get_retriever()
    res = retrieve.invoke("公司理念与行为总则第五条是什么，并且将公司每个季度的财报进行总结")
    for doc in res:
        print(doc.page_content)
        print("="*20)

