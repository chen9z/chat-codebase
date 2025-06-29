"""
连续语义分块器
确保生成的块是连续的、不重叠的
"""

import logging
import uuid
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

if TREE_SITTER_AVAILABLE:
    from src.config.tree_sitter_config import get_parser
else:
    def get_parser(language: str):
        return None

from .enhanced_models import (
    SemanticNode, NodeType, Span, 
    DefaultNodeClassifier, RelationshipAnalyzer
)
from .models import Document, SplitterConfig
from .default_splitter import DefaultSplitter


@dataclass
class ContinuousChunk:
    """连续的语义块"""
    span: Span
    nodes: List[SemanticNode]
    primary_node: Optional[SemanticNode] = None
    
    @property
    def priority(self) -> int:
        """块的优先级（最高节点的优先级）"""
        if not self.nodes:
            return 0
        return max(node.priority for node in self.nodes)
    
    @property
    def size(self) -> int:
        """块的大小"""
        return self.span.size


class ContinuousSemanticSplitter:
    """连续语义分块器"""
    
    def __init__(self, language: str, config: SplitterConfig = None):
        self.language = language
        self.config = config or SplitterConfig.for_code()
        self._fallback_splitter = DefaultSplitter(self.config)
        self._classifier = DefaultNodeClassifier()
        self._relationship_analyzer = RelationshipAnalyzer(
            max_distance_lines=self.config.max_lines_between_comment
        )
    
    def split(self, path: str, text: str) -> List[Document]:
        """分割代码为连续的语义块"""
        try:
            parser = get_parser(self.language)
            if parser is None:
                logging.warning(f"No parser available for {self.language}, falling back to default")
                return self._fallback_splitter.split(path, text)

            tree = parser.parse(bytes(text, 'utf-8'))
            root_node = tree.root_node

            if not root_node or root_node.type == "ERROR":
                logging.warning(f"Parse error for {path}, falling back to default")
                return self._fallback_splitter.split(path, text)

            # 解析语义节点
            semantic_nodes = self._parse_semantic_nodes(root_node, text)
            
            if not semantic_nodes:
                logging.info(f"No semantic nodes found for {path}, falling back to default")
                return self._fallback_splitter.split(path, text)

            # 分析节点关系
            analyzed_nodes = self._relationship_analyzer.analyze_relationships(semantic_nodes, text)

            # 创建连续的语义块
            continuous_chunks = self._create_continuous_chunks(analyzed_nodes, text)

            # 生成最终文档
            return self._create_documents(continuous_chunks, path, text)

        except Exception as e:
            logging.error(f"Continuous splitter error for {path}: {e}, falling back to default")
            return self._fallback_splitter.split(path, text)
    
    def _parse_semantic_nodes(self, root_node, text: str) -> List[SemanticNode]:
        """解析语义节点"""
        nodes = []
        
        def traverse(ast_node):
            node_type = self._classifier.classify(ast_node, self.language)
            
            # 只处理重要的节点
            if self._is_important_node(node_type, ast_node):
                metadata = self._classifier.extract_metadata(ast_node, text)
                
                semantic_node = SemanticNode(
                    node_type=node_type,
                    ast_node_type=ast_node.type,
                    span=Span(ast_node.start_byte, ast_node.end_byte),
                    name=metadata.get("name", ""),
                    content=text[ast_node.start_byte:ast_node.end_byte],
                    metadata=metadata
                )
                
                nodes.append(semantic_node)
            
            for child in ast_node.children:
                traverse(child)
        
        traverse(root_node)
        return sorted(nodes, key=lambda n: n.span.start)
    
    def _is_important_node(self, node_type: NodeType, ast_node) -> bool:
        """判断节点是否重要"""
        # 高价值节点
        high_value_types = {
            NodeType.CLASS, NodeType.INTERFACE, NodeType.FUNCTION, 
            NodeType.METHOD, NodeType.CONSTRUCTOR, NodeType.ENUM,
            NodeType.DOC_COMMENT, NodeType.IMPORT
        }
        
        if node_type in high_value_types:
            return True
        
        # 大型节点
        if ast_node.end_byte - ast_node.start_byte > 300:
            return True
        
        return False
    
    def _create_continuous_chunks(self, nodes: List[SemanticNode], text: str) -> List[ContinuousChunk]:
        """创建连续的语义块"""
        if not nodes:
            return []
        
        chunks = []
        current_chunk_nodes = []
        current_start = 0
        current_size = 0
        
        # 按位置排序确保连续性
        sorted_nodes = sorted(nodes, key=lambda n: n.span.start)
        
        for i, node in enumerate(sorted_nodes):
            node_size = node.size
            
            # 检查是否应该结束当前块
            should_end_chunk = self._should_end_current_chunk(
                current_chunk_nodes, node, current_size, node_size, text
            )
            
            if should_end_chunk and current_chunk_nodes:
                # 完成当前块
                chunk = self._finalize_chunk(current_chunk_nodes, current_start, text)
                if chunk:
                    chunks.append(chunk)
                
                # 开始新块
                current_chunk_nodes = [node]
                current_start = node.span.start
                current_size = node_size
            else:
                # 添加到当前块
                if not current_chunk_nodes:
                    current_start = node.span.start
                current_chunk_nodes.append(node)
                current_size += node_size
        
        # 处理最后一个块
        if current_chunk_nodes:
            chunk = self._finalize_chunk(current_chunk_nodes, current_start, text)
            if chunk:
                chunks.append(chunk)
        
        # 填补空隙
        filled_chunks = self._fill_gaps(chunks, text)
        
        return filled_chunks
    
    def _should_end_current_chunk(self, current_nodes: List[SemanticNode], 
                                 new_node: SemanticNode, current_size: int, 
                                 new_node_size: int, text: str) -> bool:
        """判断是否应该结束当前块"""
        if not current_nodes:
            return False
        
        # 大小限制
        if current_size + new_node_size > self.config.chunk_size:
            return True
        
        # 高优先级节点独立
        if (new_node.priority >= self.config.high_priority_threshold and
            new_node_size > self.config.chunk_size * 0.3):
            return True
        
        # 不同类型的高优先级节点分开
        primary_node = self._get_primary_node(current_nodes)
        if (primary_node and 
            primary_node.priority >= self.config.high_priority_threshold and
            new_node.priority >= self.config.high_priority_threshold and
            primary_node.node_type != new_node.node_type):
            return True
        
        # 距离太远
        last_node = current_nodes[-1]
        gap_size = new_node.span.start - last_node.span.end
        if gap_size > 1000:  # 1000字节的间隔
            return True
        
        return False
    
    def _get_primary_node(self, nodes: List[SemanticNode]) -> Optional[SemanticNode]:
        """获取节点列表中的主要节点"""
        if not nodes:
            return None
        return max(nodes, key=lambda n: n.priority)
    
    def _finalize_chunk(self, nodes: List[SemanticNode], start_pos: int, text: str) -> Optional[ContinuousChunk]:
        """完成块的创建"""
        if not nodes:
            return None
        
        # 计算块的范围
        min_start = min(node.span.start for node in nodes)
        max_end = max(node.span.end for node in nodes)
        
        # 包含相关节点的内容
        for node in nodes:
            for related_node in node.related_nodes:
                if related_node not in nodes:
                    # 检查相关节点是否在合理范围内
                    if (related_node.span.start >= min_start - 500 and 
                        related_node.span.end <= max_end + 500):
                        min_start = min(min_start, related_node.span.start)
                        max_end = max(max_end, related_node.span.end)
        
        # 按行边界对齐
        aligned_span = self._align_to_line_boundaries(Span(min_start, max_end), text)
        
        # 确定主要节点
        primary_node = self._get_primary_node(nodes)
        
        return ContinuousChunk(
            span=aligned_span,
            nodes=nodes,
            primary_node=primary_node
        )
    
    def _fill_gaps(self, chunks: List[ContinuousChunk], text: str) -> List[ContinuousChunk]:
        """填补块之间的空隙"""
        if not chunks:
            return []
        
        filled_chunks = []
        text_length = len(text)
        
        # 按位置排序
        sorted_chunks = sorted(chunks, key=lambda c: c.span.start)
        
        current_pos = 0
        
        for chunk in sorted_chunks:
            # 检查是否有空隙
            if current_pos < chunk.span.start:
                gap_content = text[current_pos:chunk.span.start].strip()
                if gap_content and len(gap_content) > 50:  # 只处理有意义的空隙
                    # 创建空隙块
                    gap_span = self._align_to_line_boundaries(
                        Span(current_pos, chunk.span.start), text
                    )
                    gap_chunk = ContinuousChunk(
                        span=gap_span,
                        nodes=[],  # 空隙块没有语义节点
                        primary_node=None
                    )
                    filled_chunks.append(gap_chunk)
            
            filled_chunks.append(chunk)
            current_pos = chunk.span.end
        
        # 处理最后的空隙
        if current_pos < text_length:
            remaining_content = text[current_pos:].strip()
            if remaining_content and len(remaining_content) > 50:
                gap_span = self._align_to_line_boundaries(
                    Span(current_pos, text_length), text
                )
                gap_chunk = ContinuousChunk(
                    span=gap_span,
                    nodes=[],
                    primary_node=None
                )
                filled_chunks.append(gap_chunk)
        
        return filled_chunks
    
    def _align_to_line_boundaries(self, span: Span, text: str) -> Span:
        """将span对齐到行边界"""
        lines = text.splitlines(keepends=True)
        total_chars = 0
        start_line_idx = 0
        end_line_idx = len(lines) - 1
        
        # 找到开始行
        for i, line in enumerate(lines):
            if total_chars + len(line) > span.start:
                start_line_idx = i
                break
            total_chars += len(line)
        
        # 找到结束行
        total_chars = 0
        for i, line in enumerate(lines):
            total_chars += len(line)
            if total_chars >= span.end:
                end_line_idx = i
                break
        
        # 计算字节位置
        start_byte_pos = sum(len(lines[i]) for i in range(start_line_idx))
        end_byte_pos = sum(len(lines[i]) for i in range(end_line_idx + 1))
        
        return Span(start_byte_pos, end_byte_pos)
    
    def _get_line_number(self, text: str, byte_index: int) -> int:
        """获取字节索引对应的行号"""
        return text[:byte_index].count('\n') + 1
    
    def _create_documents(self, chunks: List[ContinuousChunk], path: str, text: str) -> List[Document]:
        """从连续块生成文档"""
        documents = []
        
        for chunk in chunks:
            content = text[chunk.span.start:chunk.span.end]
            
            document = Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=content,
                start_line=self._get_line_number(text, chunk.span.start),
                end_line=self._get_line_number(text, chunk.span.end)
            )
            
            documents.append(document)
        
        return documents
