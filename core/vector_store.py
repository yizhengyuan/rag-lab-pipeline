"""
向量存储管理模块
"""

import os
import json
import logging
import shutil
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStore

# 先定义logger
logger = logging.getLogger(__name__)

# 添加Chroma支持
try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("Chroma未安装，将使用SimpleVectorStore")

from .nodes import ConceptNode, EvidenceNode
from utils.helpers import FileHelper

class VectorStoreManager:
    """向量存储管理器 - 使用 LlamaIndex 的向量存储系统"""
    
    def __init__(self, config):
        """
        初始化向量存储管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.store_type = config.get("vector_store.type", "simple")
        self.persist_directory = config.get("vector_store.persist_directory", "./vector_db")
        self.collection_name = config.get("vector_store.collection_name", "concepts")
        self.dimension = config.get("vector_store.dimension", 1536)
        
        # 确保持久化目录存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Chroma客户端
        self.chroma_client = None
        if self.store_type == "chroma" and CHROMA_AVAILABLE:
            self._init_chroma_client()
        
        # 存储索引
        self.chunk_index: Optional[VectorStoreIndex] = None
        self.concept_index: Optional[VectorStoreIndex] = None
        self.evidence_index: Optional[VectorStoreIndex] = None
        
        # 索引元数据
        self.index_metadata = {
            "chunks": {"created_at": None, "node_count": 0, "last_updated": None},
            "concepts": {"created_at": None, "node_count": 0, "last_updated": None},
            "evidence": {"created_at": None, "node_count": 0, "last_updated": None}
        }
        
        # 加载元数据
        self._load_metadata()
    
    def _init_chroma_client(self):
        """初始化Chroma客户端"""
        try:
            # 使用持久化客户端
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"✅ Chroma客户端初始化成功: {self.persist_directory}")
        except Exception as e:
            logger.error(f"❌ Chroma客户端初始化失败: {e}")
            logger.warning("⚠️ 将回退到SimpleVectorStore")
            self.store_type = "simple"
            self.chroma_client = None
    
    def _get_vector_store(self, collection_suffix: str = "") -> VectorStore:
        """
        获取向量存储实例
        
        Args:
            collection_suffix: 集合名称后缀，用于区分不同类型的索引
        """
        if self.store_type == "chroma" and self.chroma_client and CHROMA_AVAILABLE:
            try:
                collection_name = f"{self.collection_name}_{collection_suffix}" if collection_suffix else self.collection_name
                
                # 获取或创建集合
                chroma_collection = self.chroma_client.get_or_create_collection(collection_name)
                
                # 创建ChromaVectorStore
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                logger.info(f"✅ 使用Chroma向量存储: {collection_name}")
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ 创建Chroma向量存储失败: {e}")
                logger.warning("⚠️ 回退到SimpleVectorStore")
        
        # 回退到SimpleVectorStore
        logger.info("📝 使用SimpleVectorStore")
        return SimpleVectorStore()
    
    def create_chunk_index(self, nodes: List[TextNode], persist: bool = True) -> VectorStoreIndex:
        """
        创建文档块索引
        
        Args:
            nodes: 文本节点列表
            persist: 是否持久化存储
            
        Returns:
            VectorStoreIndex: 向量索引
        """
        logger.info(f"创建文档块索引，共 {len(nodes)} 个节点")
        
        try:
            # 获取向量存储（使用chunks后缀）
            vector_store = self._get_vector_store("chunks")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建索引
            self.chunk_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True
            )
            
            # 更新元数据
            self.index_metadata["chunks"] = {
                "created_at": datetime.now().isoformat(),
                "node_count": len(nodes),
                "last_updated": datetime.now().isoformat()
            }
            
            # 对于Chroma，数据已经自动持久化
            if self.store_type == "chroma":
                logger.info("✅ 文档块embedding已自动保存到Chroma数据库")
            elif persist:
                # 非Chroma的持久化存储
                chunk_persist_dir = os.path.join(self.persist_directory, "chunks")
                self._persist_index(self.chunk_index, chunk_persist_dir, "chunks")
            
            logger.info("文档块索引创建成功")
            return self.chunk_index
            
        except Exception as e:
            logger.error(f"创建文档块索引失败: {e}")
            raise e
    
    def create_concept_index(self, nodes: List[Union[ConceptNode, TextNode]], persist: bool = True) -> VectorStoreIndex:
        """
        创建概念索引
        
        Args:
            nodes: 概念节点列表
            persist: 是否持久化存储
            
        Returns:
            VectorStoreIndex: 概念向量索引
        """
        logger.info(f"创建概念索引，共 {len(nodes)} 个概念")
        
        try:
            # 获取向量存储（使用concepts后缀）
            vector_store = self._get_vector_store("concepts")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建索引
            self.concept_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True
            )
            
            # 更新元数据
            self.index_metadata["concepts"] = {
                "created_at": datetime.now().isoformat(),
                "node_count": len(nodes),
                "last_updated": datetime.now().isoformat()
            }
            
            # 对于Chroma，数据已经自动持久化
            if self.store_type == "chroma":
                logger.info("✅ 概念embedding已自动保存到Chroma数据库")
            elif persist:
                # 非Chroma的持久化存储
                concept_persist_dir = os.path.join(self.persist_directory, "concepts")
                self._persist_index(self.concept_index, concept_persist_dir, "concepts")
            
            logger.info("概念索引创建成功")
            return self.concept_index
            
        except Exception as e:
            logger.error(f"创建概念索引失败: {e}")
            raise e
    
    def create_evidence_index(self, nodes: List[Union[EvidenceNode, TextNode]], persist: bool = True) -> VectorStoreIndex:
        """
        创建证据索引
        
        Args:
            nodes: 证据节点列表
            persist: 是否持久化存储
            
        Returns:
            VectorStoreIndex: 证据向量索引
        """
        logger.info(f"创建证据索引，共 {len(nodes)} 个证据")
        
        try:
            # 获取向量存储（使用evidence后缀）
            vector_store = self._get_vector_store("evidence")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 创建索引
            self.evidence_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True
            )
            
            # 更新元数据
            self.index_metadata["evidence"] = {
                "created_at": datetime.now().isoformat(),
                "node_count": len(nodes),
                "last_updated": datetime.now().isoformat()
            }
            
            # 对于Chroma，数据已经自动持久化
            if self.store_type == "chroma":
                logger.info("✅ 证据embedding已自动保存到Chroma数据库")
            elif persist:
                # 非Chroma的持久化存储
                evidence_persist_dir = os.path.join(self.persist_directory, "evidence")
                self._persist_index(self.evidence_index, evidence_persist_dir, "evidence")
            
            logger.info("证据索引创建成功")
            return self.evidence_index
            
        except Exception as e:
            logger.error(f"创建证据索引失败: {e}")
            raise e
    
    def _persist_index(self, index: VectorStoreIndex, persist_dir: str, index_type: str):
        """持久化索引"""
        try:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
            
            # 保存额外的节点信息
            self._save_node_metadata(index, persist_dir, index_type)
            
            logger.info(f"{index_type}索引已持久化到: {persist_dir}")
        except Exception as e:
            logger.error(f"持久化{index_type}索引失败: {e}")
            raise e
    
    def _save_node_metadata(self, index: VectorStoreIndex, persist_dir: str, index_type: str):
        """保存节点元数据"""
        try:
            nodes_info = []
            for node_id, node in index.docstore.docs.items():
                node_info = {
                    "node_id": node_id,
                    "text_preview": node.text[:200] if hasattr(node, 'text') else "",
                    "metadata": node.metadata if hasattr(node, 'metadata') else {},
                    "node_type": getattr(node, 'node_type', 'unknown')
                }
                nodes_info.append(node_info)
            
            metadata_file = os.path.join(persist_dir, f"{index_type}_metadata.json")
            FileHelper.save_json(nodes_info, metadata_file)
            
        except Exception as e:
            logger.warning(f"保存{index_type}节点元数据失败: {e}")
    
    def load_chunk_index(self) -> Optional[VectorStoreIndex]:
        """
        加载已持久化的文档块索引
        
        Returns:
            Optional[VectorStoreIndex]: 加载的索引，如果不存在则返回None
        """
        return self._load_index("chunks", "chunk_index")
    
    def load_concept_index(self) -> Optional[VectorStoreIndex]:
        """
        加载已持久化的概念索引
        
        Returns:
            Optional[VectorStoreIndex]: 加载的索引，如果不存在则返回None
        """
        return self._load_index("concepts", "concept_index")
    
    def load_evidence_index(self) -> Optional[VectorStoreIndex]:
        """
        加载已持久化的证据索引
        
        Returns:
            Optional[VectorStoreIndex]: 加载的索引，如果不存在则返回None
        """
        return self._load_index("evidence", "evidence_index")
    
    def _load_index(self, index_name: str, attr_name: str) -> Optional[VectorStoreIndex]:
        """通用的索引加载方法"""
        persist_dir = os.path.join(self.persist_directory, index_name)
        
        if not os.path.exists(persist_dir):
            logger.info(f"未找到持久化的{index_name}索引")
            return None
        
        try:
            # 重建存储上下文
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            
            # 加载索引
            index = VectorStoreIndex.from_storage_context(storage_context)
            setattr(self, attr_name, index)
            
            logger.info(f"成功加载{index_name}索引: {persist_dir}")
            return index
            
        except Exception as e:
            logger.warning(f"加载{index_name}索引失败: {e}")
            return None
    
    def update_index(self, index_type: str, new_nodes: List[BaseNode]) -> bool:
        """
        更新索引（添加新节点）
        
        Args:
            index_type: 索引类型 ("chunks", "concepts", "evidence")
            new_nodes: 新节点列表
            
        Returns:
            bool: 是否成功更新
        """
        try:
            if index_type == "chunks" and self.chunk_index:
                for node in new_nodes:
                    self.chunk_index.insert(node)
                self.index_metadata["chunks"]["last_updated"] = datetime.now().isoformat()
                self.index_metadata["chunks"]["node_count"] += len(new_nodes)
                
            elif index_type == "concepts" and self.concept_index:
                for node in new_nodes:
                    self.concept_index.insert(node)
                self.index_metadata["concepts"]["last_updated"] = datetime.now().isoformat()
                self.index_metadata["concepts"]["node_count"] += len(new_nodes)
                
            elif index_type == "evidence" and self.evidence_index:
                for node in new_nodes:
                    self.evidence_index.insert(node)
                self.index_metadata["evidence"]["last_updated"] = datetime.now().isoformat()
                self.index_metadata["evidence"]["node_count"] += len(new_nodes)
            else:
                logger.error(f"无效的索引类型或索引不存在: {index_type}")
                return False
            
            # 保存元数据
            self._save_metadata()
            
            logger.info(f"成功更新{index_type}索引，添加了{len(new_nodes)}个节点")
            return True
            
        except Exception as e:
            logger.error(f"更新{index_type}索引失败: {e}")
            return False
    
    def delete_nodes(self, index_type: str, node_ids: List[str]) -> bool:
        """
        从索引中删除节点
        
        Args:
            index_type: 索引类型
            node_ids: 要删除的节点ID列表
            
        Returns:
            bool: 是否成功删除
        """
        try:
            index = getattr(self, f"{index_type}_index", None)
            if not index:
                logger.error(f"{index_type}索引不存在")
                return False
            
            for node_id in node_ids:
                index.delete(node_id)
            
            # 更新元数据
            self.index_metadata[index_type]["last_updated"] = datetime.now().isoformat()
            self.index_metadata[index_type]["node_count"] -= len(node_ids)
            self._save_metadata()
            
            logger.info(f"成功从{index_type}索引删除{len(node_ids)}个节点")
            return True
            
        except Exception as e:
            logger.error(f"删除{index_type}索引节点失败: {e}")
            return False
    
    def clear_indexes(self):
        """清除所有索引"""
        self.chunk_index = None
        self.concept_index = None
        self.evidence_index = None
        
        # 重置元数据
        for index_type in self.index_metadata:
            self.index_metadata[index_type] = {
                "created_at": None, 
                "node_count": 0, 
                "last_updated": None
            }
        
        logger.info("已清除所有索引")
    
    def delete_persisted_indexes(self, index_types: List[str] = None):
        """
        删除持久化的索引文件
        
        Args:
            index_types: 要删除的索引类型列表，None表示删除所有
        """
        if index_types is None:
            index_types = ["chunks", "concepts", "evidence"]
        
        for index_type in index_types:
            persist_dir = os.path.join(self.persist_directory, index_type)
            try:
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)
                    logger.info(f"已删除{index_type}索引: {persist_dir}")
                    
                    # 重置元数据
                    self.index_metadata[index_type] = {
                        "created_at": None, 
                        "node_count": 0, 
                        "last_updated": None
                    }
                    
            except Exception as e:
                logger.error(f"删除{index_type}索引失败: {e}")
        
        self._save_metadata()
    
    def backup_indexes(self, backup_dir: str) -> bool:
        """
        备份索引
        
        Args:
            backup_dir: 备份目录
            
        Returns:
            bool: 是否成功备份
        """
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 创建时间戳目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_backup = backup_path / f"backup_{timestamp}"
            timestamped_backup.mkdir(exist_ok=True)
            
            # 复制索引目录
            for index_type in ["chunks", "concepts", "evidence"]:
                source_dir = os.path.join(self.persist_directory, index_type)
                if os.path.exists(source_dir):
                    target_dir = timestamped_backup / index_type
                    shutil.copytree(source_dir, target_dir)
                    logger.info(f"已备份{index_type}索引到: {target_dir}")
            
            # 备份元数据
            metadata_backup = timestamped_backup / "metadata.json"
            FileHelper.save_json(self.index_metadata, str(metadata_backup))
            
            logger.info(f"索引备份完成: {timestamped_backup}")
            return True
            
        except Exception as e:
            logger.error(f"备份索引失败: {e}")
            return False
    
    def restore_indexes(self, backup_dir: str) -> bool:
        """
        从备份恢复索引
        
        Args:
            backup_dir: 备份目录
            
        Returns:
            bool: 是否成功恢复
        """
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                logger.error(f"备份目录不存在: {backup_dir}")
                return False
            
            # 清除当前索引
            self.clear_indexes()
            self.delete_persisted_indexes()
            
            # 恢复索引目录
            for index_type in ["chunks", "concepts", "evidence"]:
                source_dir = backup_path / index_type
                if source_dir.exists():
                    target_dir = Path(self.persist_directory) / index_type
                    shutil.copytree(source_dir, target_dir)
                    logger.info(f"已恢复{index_type}索引从: {source_dir}")
            
            # 恢复元数据
            metadata_backup = backup_path / "metadata.json"
            if metadata_backup.exists():
                self.index_metadata = FileHelper.load_json(str(metadata_backup))
            
            # 重新加载索引
            self.rebuild_indexes_from_persisted()
            
            logger.info(f"索引恢复完成: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"恢复索引失败: {e}")
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息
        
        Returns:
            Dict[str, Any]: 索引信息
        """
        info = {
            "store_type": self.store_type,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "indexes": {}
        }
        
        # 添加各个索引的信息
        for index_type in ["chunks", "concepts", "evidence"]:
            index = getattr(self, f"{index_type}_index", None)
            persist_dir = os.path.join(self.persist_directory, index_type)
            
            index_info = {
                "exists": index is not None,
                "persisted": os.path.exists(persist_dir),
                "metadata": self.index_metadata.get(index_type, {})
            }
            
            # 尝试获取实际节点数量
            if index:
                try:
                    index_info["actual_node_count"] = len(index.docstore.docs)
                except:
                    index_info["actual_node_count"] = "unknown"
            
            info["indexes"][index_type] = index_info
        
        return info
    
    def get_storage_size(self) -> Dict[str, Any]:
        """获取存储大小信息"""
        def get_dir_size(path):
            total = 0
            try:
                for entry in os.scandir(path):
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += get_dir_size(entry.path)
            except:
                pass
            return total
        
        size_info = {}
        total_size = 0
        
        for index_type in ["chunks", "concepts", "evidence"]:
            persist_dir = os.path.join(self.persist_directory, index_type)
            if os.path.exists(persist_dir):
                size = get_dir_size(persist_dir)
                size_info[index_type] = {
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                }
                total_size += size
            else:
                size_info[index_type] = {"size_bytes": 0, "size_mb": 0}
        
        size_info["total"] = {
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }
        
        return size_info
    
    def rebuild_indexes_from_persisted(self) -> bool:
        """
        从持久化文件重建索引
        
        Returns:
            bool: 是否成功重建
        """
        success_count = 0
        
        # 尝试加载各个索引
        if self.load_chunk_index():
            success_count += 1
        
        if self.load_concept_index():
            success_count += 1
            
        if self.load_evidence_index():
            success_count += 1
        
        if success_count > 0:
            logger.info(f"成功重建{success_count}个索引")
            return True
        else:
            logger.warning("没有成功重建任何索引")
            return False
    
    def _load_metadata(self):
        """加载元数据"""
        metadata_file = os.path.join(self.persist_directory, "index_metadata.json")
        if os.path.exists(metadata_file):
            try:
                self.index_metadata = FileHelper.load_json(metadata_file)
                logger.debug("成功加载索引元数据")
            except Exception as e:
                logger.warning(f"加载索引元数据失败: {e}")
    
    def _save_metadata(self):
        """保存元数据"""
        metadata_file = os.path.join(self.persist_directory, "index_metadata.json")
        try:
            FileHelper.save_json(self.index_metadata, metadata_file)
            logger.debug("成功保存索引元数据")
        except Exception as e:
            logger.warning(f"保存索引元数据失败: {e}")
    
    def optimize_indexes(self) -> bool:
        """
        优化索引（重建以提高性能）
        
        Returns:
            bool: 是否成功优化
        """
        try:
            logger.info("开始优化索引...")
            
            # 备份当前索引
            backup_success = self.backup_indexes(
                os.path.join(self.persist_directory, "optimization_backup")
            )
            
            if not backup_success:
                logger.error("备份失败，取消优化")
                return False
            
            # 收集所有节点
            all_nodes = {"chunks": [], "concepts": [], "evidence": []}
            
            for index_type in ["chunks", "concepts", "evidence"]:
                index = getattr(self, f"{index_type}_index", None)
                if index:
                    nodes = list(index.docstore.docs.values())
                    all_nodes[index_type] = nodes
            
            # 清除当前索引
            self.clear_indexes()
            self.delete_persisted_indexes()
            
            # 重建索引
            rebuild_success = True
            for index_type, nodes in all_nodes.items():
                if nodes:
                    try:
                        if index_type == "chunks":
                            self.create_chunk_index(nodes)
                        elif index_type == "concepts":
                            self.create_concept_index(nodes)
                        elif index_type == "evidence":
                            self.create_evidence_index(nodes)
                    except Exception as e:
                        logger.error(f"重建{index_type}索引失败: {e}")
                        rebuild_success = False
            
            if rebuild_success:
                logger.info("索引优化完成")
                # 删除备份
                backup_dir = os.path.join(self.persist_directory, "optimization_backup")
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                return True
            else:
                logger.error("索引优化失败，尝试恢复备份")
                return self.restore_indexes(
                    os.path.join(self.persist_directory, "optimization_backup")
                )
                
        except Exception as e:
            logger.error(f"索引优化过程中出错: {e}")
            return False 