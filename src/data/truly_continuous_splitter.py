"""
真正连续的语义分块器
确保生成的块严格连续、无重叠、全覆盖
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
class ChunkBoundary:
    """块边界点"""
    position: int  # 字节位置
    priority: int  # 优先级
    reason: str    # 边界原因
    node: Optional[SemanticNode] = None


class TrulyContinuousSplitter:
    """真正连续的语义分块器"""
    
    def __init__(self, language: str, config: SplitterConfig = None):
        self.language = language
        self.config = config or SplitterConfig.for_code()
        self._fallback_splitter = DefaultSplitter(self.config)
        self._classifier = DefaultNodeClassifier()
        self._relationship_analyzer = RelationshipAnalyzer(
            max_distance_lines=self.config.max_lines_between_comment
        )
    
    def split(self, path: str, text: str) -> List[Document]:
        """分割代码为真正连续的语义块"""
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

            # 确定分块边界
            boundaries = self._determine_chunk_boundaries(analyzed_nodes, text)

            # 创建连续块
            chunks = self._create_continuous_chunks(boundaries, text)

            # 生成最终文档
            return self._create_documents(chunks, path, text)

        except Exception as e:
            logging.error(f"Truly continuous splitter error for {path}: {e}, falling back to default")
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
    
    def _determine_chunk_boundaries(self, nodes: List[SemanticNode], text: str) -> List[ChunkBoundary]:
        """确定分块边界"""
        boundaries = []
        
        # 添加文件开始边界
        boundaries.append(ChunkBoundary(0, 0, "file_start"))
        
        # 基于语义节点确定边界
        for i, node in enumerate(nodes):
            # 高优先级节点的开始位置作为边界
            if node.priority >= self.config.high_priority_threshold:
                # 检查是否应该在此节点前分割
                if self._should_split_before_node(node, nodes[:i], text):
                    boundary_pos = self._find_optimal_boundary_before(node, text)
                    boundaries.append(ChunkBoundary(
                        boundary_pos, 
                        node.priority, 
                        f"before_high_priority_{node.node_type.type_name}",
                        node
                    ))
        
        # 基于大小限制确定边界
        current_pos = 0
        current_size = 0
        
        for boundary in sorted(boundaries, key=lambda b: b.position):
            if boundary.position > current_pos:
                segment_size = boundary.position - current_pos
                
                if current_size + segment_size > self.config.chunk_size:
                    # 需要在中间插入边界
                    split_pos = current_pos + (self.config.chunk_size - current_size)
                    optimal_pos = self._find_optimal_boundary_near(split_pos, text)
                    
                    boundaries.append(ChunkBoundary(
                        optimal_pos,
                        1,
                        "size_limit"
                    ))
                    current_pos = optimal_pos
                    current_size = 0
                else:
                    current_size += segment_size
                    current_pos = boundary.position
        
        # 添加文件结束边界
        boundaries.append(ChunkBoundary(len(text), 0, "file_end"))
        
        # 去重并排序
        unique_boundaries = []
        seen_positions = set()
        
        for boundary in sorted(boundaries, key=lambda b: b.position):
            if boundary.position not in seen_positions:
                unique_boundaries.append(boundary)
                seen_positions.add(boundary.position)
        
        return unique_boundaries
    
    def _should_split_before_node(self, node: SemanticNode, prev_nodes: List[SemanticNode], text: str) -> bool:
        """判断是否应该在节点前分割"""
        # 类和接口总是分割
        if node.node_type in [NodeType.CLASS, NodeType.INTERFACE]:
            return True
        
        # 大型函数分割
        if node.node_type == NodeType.FUNCTION and node.size > self.config.chunk_size * 0.4:
            return True
        
        # 与前面节点类型不同的高优先级节点
        if prev_nodes:
            last_node = prev_nodes[-1]
            if (node.priority >= self.config.high_priority_threshold and
                last_node.priority >= self.config.high_priority_threshold and
                node.node_type != last_node.node_type):
                return True
        
        return False
    
    def _find_optimal_boundary_before(self, node: SemanticNode, text: str) -> int:
        """在节点前找到最佳边界位置"""
        target_pos = node.span.start
        
        # 查找相关注释
        for related_node in node.related_nodes:
            if (related_node.node_type.category == "documentation" and
                related_node.span.end <= target_pos):
                target_pos = min(target_pos, related_node.span.start)
        
        # 对齐到行边界
        return self._align_to_line_start(target_pos, text)
    
    def _find_optimal_boundary_near(self, target_pos: int, text: str) -> int:
        """在目标位置附近找到最佳边界"""
        # 在目标位置前后寻找合适的分割点
        search_range = 200  # 200字节搜索范围
        
        start_search = max(0, target_pos - search_range)
        end_search = min(len(text), target_pos + search_range)
        
        # 寻找行边界
        best_pos = target_pos
        best_score = 0
        
        for pos in range(start_search, end_search):
            if pos < len(text) and text[pos] == '\n':
                # 计算分割点的质量分数
                score = self._calculate_boundary_score(pos + 1, target_pos, text)
                if score > best_score:
                    best_score = score
                    best_pos = pos + 1
        
        return best_pos
    
    def _calculate_boundary_score(self, pos: int, target_pos: int, text: str) -> float:
        """计算边界位置的质量分数"""
        score = 0.0
        
        # 距离目标位置越近越好
        distance = abs(pos - target_pos)
        score += max(0, 100 - distance / 10)
        
        # 检查前后内容
        if pos > 0 and pos < len(text):
            # 前面是空行或注释结束
            prev_line = self._get_line_at_pos(pos - 1, text).strip()
            if not prev_line or prev_line.startswith(('#', '//', '*')):
                score += 20
            
            # 后面是新的语义单元开始
            next_line = self._get_line_at_pos(pos, text).strip()
            if (next_line.startswith(('class ', 'def ', 'function ', 'public ', 'private ')) or
                next_line.startswith(('import ', 'from ', '#', '//', '/*'))):
                score += 30
        
        return score
    
    def _get_line_at_pos(self, pos: int, text: str) -> str:
        """获取指定位置所在的行"""
        if pos < 0 or pos >= len(text):
            return ""
        
        # 找到行的开始
        line_start = pos
        while line_start > 0 and text[line_start - 1] != '\n':
            line_start -= 1
        
        # 找到行的结束
        line_end = pos
        while line_end < len(text) and text[line_end] != '\n':
            line_end += 1
        
        return text[line_start:line_end]
    
    def _align_to_line_start(self, pos: int, text: str) -> int:
        """将位置对齐到行开始"""
        if pos <= 0:
            return 0
        
        # 向前找到行开始
        while pos > 0 and text[pos - 1] != '\n':
            pos -= 1
        
        return pos
    
    def _create_continuous_chunks(self, boundaries: List[ChunkBoundary], text: str) -> List[Span]:
        """根据边界创建连续块"""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_pos = boundaries[i].position
            end_pos = boundaries[i + 1].position
            
            if end_pos > start_pos:
                # 确保边界对齐到行
                aligned_start = self._align_to_line_start(start_pos, text)
                aligned_end = self._align_to_line_start(end_pos, text)
                
                # 如果对齐后位置相同，调整结束位置
                if aligned_end <= aligned_start:
                    aligned_end = min(len(text), end_pos)
                
                if aligned_end > aligned_start:
                    chunks.append(Span(aligned_start, aligned_end))
        
        return chunks
    
    def _create_documents(self, chunks: List[Span], path: str, text: str) -> List[Document]:
        """从连续块生成文档"""
        documents = []
        
        for chunk in chunks:
            content = text[chunk.start:chunk.end]
            
            # 跳过空块
            if not content.strip():
                continue
            
            document = Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=content,
                start_line=self._get_line_number(text, chunk.start),
                end_line=self._get_line_number(text, chunk.end - 1)
            )
            
            documents.append(document)
        
        return documents
    
    def _get_line_number(self, text: str, byte_index: int) -> int:
        """获取字节索引对应的行号"""
        if byte_index < 0:
            return 1
        if byte_index >= len(text):
            return len(text.splitlines()) or 1
        return text[:byte_index].count('\n') + 1
