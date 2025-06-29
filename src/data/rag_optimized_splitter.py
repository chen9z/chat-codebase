"""
RAG 优化的语义分块器
专门为 Codebase RAG 应用设计，优化检索和生成效果
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

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
    SemanticNode, NodeType, Span, NodeGroup, 
    DefaultNodeClassifier, RelationshipAnalyzer
)
from .models import Document, SplitterConfig
from .default_splitter import DefaultSplitter


@dataclass
class RAGChunk:
    """RAG 优化的代码块"""
    
    # 基本信息
    chunk_id: str
    path: str
    content: str
    start_line: int
    end_line: int
    
    # RAG 特定的元数据
    primary_element: str = ""           # 主要代码元素（类名、函数名等）
    element_type: str = ""              # 元素类型（class, function, method等）
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他元素
    exports: List[str] = field(default_factory=list)       # 导出的元素
    
    # 检索优化
    keywords: List[str] = field(default_factory=list)      # 关键词
    summary: str = ""                   # 代码块摘要
    complexity_score: float = 0.0       # 复杂度评分
    
    # 上下文信息
    context_before: str = ""            # 前置上下文
    context_after: str = ""             # 后置上下文
    related_chunks: List[str] = field(default_factory=list)  # 相关块ID
    
    def to_document(self) -> Document:
        """转换为标准Document格式"""
        return Document(
            chunk_id=self.chunk_id,
            path=self.path,
            content=self.content,
            start_line=self.start_line,
            end_line=self.end_line
        )


class RAGOptimizedSplitter:
    """RAG 优化的语义分割器"""
    
    def __init__(self, language: str, config: SplitterConfig = None):
        self.language = language
        self.config = config or SplitterConfig.for_code()
        self._fallback_splitter = DefaultSplitter(self.config)
        self._classifier = DefaultNodeClassifier()
        self._relationship_analyzer = RelationshipAnalyzer(
            max_distance_lines=self.config.max_lines_between_comment
        )
    
    def split(self, path: str, text: str) -> List[Document]:
        """分割代码为 RAG 优化的块"""
        try:
            # 生成 RAG 优化的块
            rag_chunks = self._create_rag_chunks(path, text)
            
            # 转换为标准 Document 格式
            return [chunk.to_document() for chunk in rag_chunks]
            
        except Exception as e:
            logging.error(f"RAG splitter error for {path}: {e}, falling back to default")
            return self._fallback_splitter.split(path, text)
    
    def _create_rag_chunks(self, path: str, text: str) -> List[RAGChunk]:
        """创建 RAG 优化的代码块"""
        parser = get_parser(self.language)
        if parser is None:
            # 降级到基于规则的分割
            return self._rule_based_rag_split(path, text)
        
        tree = parser.parse(bytes(text, 'utf-8'))
        root_node = tree.root_node
        
        if not root_node or root_node.type == "ERROR":
            return self._rule_based_rag_split(path, text)
        
        # 解析语义节点
        semantic_nodes = self._parse_semantic_nodes(root_node, text)
        
        # 分析节点关系
        analyzed_nodes = self._relationship_analyzer.analyze_relationships(semantic_nodes, text)
        
        # 创建 RAG 优化的分组
        rag_groups = self._create_rag_groups(analyzed_nodes, text)
        
        # 生成最终的 RAG 块
        return self._generate_rag_chunks(rag_groups, path, text)
    
    def _parse_semantic_nodes(self, root_node, text: str) -> List[SemanticNode]:
        """解析语义节点，专注于 RAG 重要的元素"""
        nodes = []
        
        def traverse(ast_node):
            node_type = self._classifier.classify(ast_node, self.language)
            
            # RAG 关注的节点类型
            if self._is_rag_important(node_type, ast_node):
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
    
    def _is_rag_important(self, node_type: NodeType, ast_node) -> bool:
        """判断节点是否对 RAG 重要"""
        # 高价值的代码元素
        high_value_types = {
            NodeType.CLASS, NodeType.INTERFACE, NodeType.FUNCTION, 
            NodeType.METHOD, NodeType.CONSTRUCTOR, NodeType.ENUM
        }
        
        if node_type in high_value_types:
            return True
        
        # 重要的文档注释
        if node_type == NodeType.DOC_COMMENT:
            return True
        
        # 大型的其他节点
        if ast_node.end_byte - ast_node.start_byte > 200:
            return True
        
        return False
    
    def _create_rag_groups(self, nodes: List[SemanticNode], text: str) -> List[NodeGroup]:
        """创建 RAG 优化的节点分组"""
        groups = []
        
        # 按功能单元分组
        functional_groups = self._group_by_functionality(nodes, text)
        
        # 优化组大小以适应 RAG
        optimized_groups = self._optimize_for_rag(functional_groups, text)
        
        return optimized_groups
    
    def _group_by_functionality(self, nodes: List[SemanticNode], text: str) -> List[NodeGroup]:
        """按功能单元分组"""
        groups = []
        current_group = NodeGroup()
        
        for node in nodes:
            # 检查是否应该开始新组
            if self._should_start_new_functional_group(node, current_group, text):
                if current_group.nodes:
                    groups.append(current_group)
                current_group = NodeGroup()
            
            current_group.add_node(node)
            
            # 添加相关节点
            for related_node in node.related_nodes:
                if related_node not in current_group.nodes:
                    current_group.add_node(related_node)
        
        if current_group.nodes:
            groups.append(current_group)
        
        return groups
    
    def _should_start_new_functional_group(self, node: SemanticNode, 
                                         current_group: NodeGroup, text: str) -> bool:
        """判断是否应该开始新的功能组"""
        if not current_group.nodes:
            return False
        
        # 大小限制
        if current_group.size + node.size > self.config.chunk_size:
            return True
        
        # 功能边界：不同的类应该分开
        if (node.node_type == NodeType.CLASS and 
            current_group.primary_node and 
            current_group.primary_node.node_type == NodeType.CLASS):
            return True
        
        # 距离太远
        if current_group.span:
            distance = node.span.start - current_group.span.end
            if distance > 1000:  # 1000字节
                return True
        
        return False
    
    def _optimize_for_rag(self, groups: List[NodeGroup], text: str) -> List[NodeGroup]:
        """为 RAG 优化分组"""
        optimized = []
        
        for group in groups:
            # 太小的组：尝试合并
            if group.size < self.config.chunk_size * 0.3:
                merged = False
                for existing_group in optimized:
                    if (existing_group.size + group.size <= self.config.chunk_size and
                        self._can_merge_for_rag(existing_group, group, text)):
                        # 合并组
                        for node in group.nodes:
                            existing_group.add_node(node)
                        merged = True
                        break
                
                if not merged:
                    optimized.append(group)
            
            # 太大的组：拆分
            elif group.size > self.config.chunk_size * 1.2:
                split_groups = self._split_for_rag(group, text)
                optimized.extend(split_groups)
            
            else:
                optimized.append(group)
        
        return optimized
    
    def _can_merge_for_rag(self, group1: NodeGroup, group2: NodeGroup, text: str) -> bool:
        """判断两个组是否可以为 RAG 合并"""
        # 相同类型的元素容易合并
        if (group1.primary_node and group2.primary_node and
            group1.primary_node.node_type == group2.primary_node.node_type):
            return True
        
        # 相关的元素可以合并
        for node1 in group1.nodes:
            for node2 in group2.nodes:
                if node2 in node1.related_nodes or node1 in node2.related_nodes:
                    return True
        
        return False
    
    def _split_for_rag(self, group: NodeGroup, text: str) -> List[NodeGroup]:
        """为 RAG 拆分大组"""
        if len(group.nodes) <= 1:
            return [group]
        
        # 按节点类型和位置重新分组
        sorted_nodes = sorted(group.nodes, key=lambda n: (n.node_type.priority, n.span.start))
        
        split_groups = []
        current_group = NodeGroup()
        
        for node in sorted_nodes:
            if current_group.size + node.size > self.config.chunk_size and current_group.nodes:
                split_groups.append(current_group)
                current_group = NodeGroup()
            
            current_group.add_node(node)
        
        if current_group.nodes:
            split_groups.append(current_group)
        
        return split_groups if split_groups else [group]
    
    def _generate_rag_chunks(self, groups: List[NodeGroup], path: str, text: str) -> List[RAGChunk]:
        """生成 RAG 优化的代码块"""
        rag_chunks = []
        
        for i, group in enumerate(groups):
            if not group.span:
                continue
            
            # 按行边界对齐
            aligned_span = self._align_to_line_boundaries(group.span, text)
            content = text[aligned_span.start:aligned_span.end]
            
            # 提取 RAG 元数据
            metadata = self._extract_rag_metadata(group, text)
            
            # 生成上下文
            context_before, context_after = self._generate_context(aligned_span, text)
            
            rag_chunk = RAGChunk(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=content,
                start_line=self._get_line_number(text, aligned_span.start),
                end_line=self._get_line_number(text, aligned_span.end),
                primary_element=metadata.get("primary_element", ""),
                element_type=metadata.get("element_type", ""),
                dependencies=metadata.get("dependencies", []),
                exports=metadata.get("exports", []),
                keywords=metadata.get("keywords", []),
                summary=metadata.get("summary", ""),
                complexity_score=metadata.get("complexity_score", 0.0),
                context_before=context_before,
                context_after=context_after
            )
            
            rag_chunks.append(rag_chunk)
        
        # 建立块之间的关系
        self._establish_chunk_relationships(rag_chunks)
        
        return rag_chunks
    
    def _extract_rag_metadata(self, group: NodeGroup, text: str) -> Dict[str, Any]:
        """提取 RAG 相关的元数据"""
        metadata = {}
        
        # 主要元素
        if group.primary_node:
            metadata["primary_element"] = group.primary_node.name or "unnamed"
            metadata["element_type"] = group.primary_node.node_type.type_name
        
        # 依赖和导出
        dependencies = set()
        exports = set()
        keywords = set()
        
        for node in group.nodes:
            # 提取关键词
            if node.name:
                keywords.add(node.name)
            
            # 根据节点类型提取信息
            if node.node_type == NodeType.IMPORT:
                # 导入的依赖
                import_names = self._extract_import_names(node.content)
                dependencies.update(import_names)
            elif node.node_type in [NodeType.CLASS, NodeType.FUNCTION]:
                # 导出的元素
                if node.name:
                    exports.add(node.name)
        
        metadata["dependencies"] = list(dependencies)
        metadata["exports"] = list(exports)
        metadata["keywords"] = list(keywords)
        
        # 生成摘要
        metadata["summary"] = self._generate_summary(group)
        
        # 计算复杂度
        metadata["complexity_score"] = self._calculate_complexity(group)
        
        return metadata
    
    def _extract_import_names(self, import_content: str) -> List[str]:
        """从导入语句中提取名称"""
        # 简单的导入名称提取
        names = []
        lines = import_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # 提取导入的模块名
                parts = line.split()
                if len(parts) >= 2:
                    names.append(parts[1].split('.')[0])
        return names
    
    def _generate_summary(self, group: NodeGroup) -> str:
        """生成代码块摘要"""
        if not group.primary_node:
            return "Code block"
        
        primary = group.primary_node
        element_type = primary.node_type.type_name
        name = primary.name or "unnamed"
        
        # 统计其他元素
        other_elements = []
        for node in group.nodes:
            if node != primary and node.node_type != NodeType.DOC_COMMENT:
                other_elements.append(node.node_type.type_name)
        
        summary = f"{element_type.title()} '{name}'"
        if other_elements:
            summary += f" with {len(other_elements)} related elements"
        
        return summary
    
    def _calculate_complexity(self, group: NodeGroup) -> float:
        """计算代码块复杂度"""
        complexity = 0.0
        
        for node in group.nodes:
            # 基于节点类型的复杂度
            type_complexity = {
                NodeType.CLASS: 3.0,
                NodeType.FUNCTION: 2.0,
                NodeType.METHOD: 2.0,
                NodeType.CONSTRUCTOR: 1.5,
                NodeType.INTERFACE: 2.5,
                NodeType.ENUM: 1.0
            }.get(node.node_type, 0.5)
            
            # 基于大小的复杂度
            size_complexity = min(node.size / 1000, 2.0)
            
            complexity += type_complexity + size_complexity
        
        return min(complexity, 10.0)  # 限制在10以内
    
    def _generate_context(self, span: Span, text: str) -> tuple[str, str]:
        """生成前后上下文"""
        context_size = 200  # 200字符的上下文
        
        # 前置上下文
        context_before_start = max(0, span.start - context_size)
        context_before = text[context_before_start:span.start].strip()
        
        # 后置上下文
        context_after_end = min(len(text), span.end + context_size)
        context_after = text[span.end:context_after_end].strip()
        
        return context_before, context_after
    
    def _establish_chunk_relationships(self, chunks: List[RAGChunk]):
        """建立块之间的关系"""
        for i, chunk in enumerate(chunks):
            related_ids = []
            
            # 查找相关的块
            for j, other_chunk in enumerate(chunks):
                if i != j and self._are_chunks_related(chunk, other_chunk):
                    related_ids.append(other_chunk.chunk_id)
            
            chunk.related_chunks = related_ids
    
    def _are_chunks_related(self, chunk1: RAGChunk, chunk2: RAGChunk) -> bool:
        """判断两个块是否相关"""
        # 依赖关系
        if (set(chunk1.exports) & set(chunk2.dependencies) or
            set(chunk2.exports) & set(chunk1.dependencies)):
            return True
        
        # 相同的主要元素类型
        if chunk1.element_type == chunk2.element_type:
            return True
        
        # 关键词重叠
        common_keywords = set(chunk1.keywords) & set(chunk2.keywords)
        if len(common_keywords) >= 2:
            return True
        
        return False
    
    def _rule_based_rag_split(self, path: str, text: str) -> List[RAGChunk]:
        """基于规则的 RAG 分割（降级方案）"""
        # 使用默认分割器
        documents = self._fallback_splitter.split(path, text)
        
        # 转换为 RAG 块
        rag_chunks = []
        for doc in documents:
            rag_chunk = RAGChunk(
                chunk_id=doc.chunk_id,
                path=doc.path,
                content=doc.content,
                start_line=doc.start_line,
                end_line=doc.end_line,
                summary=f"Code block (lines {doc.start_line}-{doc.end_line})"
            )
            rag_chunks.append(rag_chunk)
        
        return rag_chunks
    
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
