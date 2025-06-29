"""
增强的语义分割器
采用通用节点解析 + 优先级合并的架构
"""

import logging
import uuid
from typing import List, Optional

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


class EnhancedSemanticSplitter:
    """增强的语义分割器，支持通用节点解析"""
    
    def __init__(self, language: str, config: SplitterConfig = None):
        self.language = language
        self.config = config or SplitterConfig.for_code()
        self._fallback_splitter = DefaultSplitter(self.config)
        self._classifier = DefaultNodeClassifier()
        self._relationship_analyzer = RelationshipAnalyzer(
            max_distance_lines=self.config.max_lines_between_comment
        )
    
    def split(self, path: str, text: str) -> List[Document]:
        """分割代码文本为语义块"""
        try:
            parser = get_parser(self.language)
            if parser is None:
                logging.warning(f"No parser available for language: {self.language}, falling back to default splitter")
                return self._fallback_splitter.split(path, text)

            tree = parser.parse(bytes(text, 'utf-8'))
            root_node = tree.root_node

            if not root_node or root_node.type == "ERROR":
                logging.warning(f"Parse error for {path}, falling back to default splitter")
                return self._fallback_splitter.split(path, text)

            # 阶段1: 解析所有节点
            semantic_nodes = self._parse_all_nodes(root_node, text)

            if not semantic_nodes:
                logging.info(f"No semantic nodes found for {path}, falling back to default splitter")
                return self._fallback_splitter.split(path, text)

            # 阶段2: 分析节点关系
            analyzed_nodes = self._relationship_analyzer.analyze_relationships(semantic_nodes, text)

            # 阶段3: 按优先级分组
            node_groups = self._group_by_priority(analyzed_nodes, text)

            # 阶段4: 生成最终文档
            return self._create_documents(node_groups, path, text)

        except Exception as e:
            logging.error(f"Unexpected error parsing {path}: {e}, falling back to default splitter")
            return self._fallback_splitter.split(path, text)
    
    def _parse_all_nodes(self, root_node, text: str) -> List[SemanticNode]:
        """解析所有有意义的节点"""
        nodes = []
        
        def traverse(ast_node):
            # 分类节点
            node_type = self._classifier.classify(ast_node, self.language)
            
            # 只处理有意义的节点（排除OTHER类型的小节点）
            if node_type != NodeType.OTHER or self._is_significant_node(ast_node):
                # 提取元数据
                metadata = self._classifier.extract_metadata(ast_node, text)
                
                # 创建语义节点
                semantic_node = SemanticNode(
                    node_type=node_type,
                    ast_node_type=ast_node.type,
                    span=Span(ast_node.start_byte, ast_node.end_byte),
                    name=metadata.get("name", ""),
                    content=text[ast_node.start_byte:ast_node.end_byte],
                    metadata=metadata
                )
                
                nodes.append(semantic_node)
            
            # 继续遍历子节点
            for child in ast_node.children:
                traverse(child)
        
        traverse(root_node)
        return sorted(nodes, key=lambda n: n.span.start)
    
    def _is_significant_node(self, ast_node) -> bool:
        """判断是否是重要的OTHER类型节点"""
        # 大型节点可能包含重要内容
        node_size = ast_node.end_byte - ast_node.start_byte
        if node_size > 500:  # 大于500字节的节点
            return True
        
        # 特定类型的节点
        significant_types = {
            "expression_statement", "if_statement", "for_statement", 
            "while_statement", "try_statement", "with_statement"
        }
        return ast_node.type in significant_types
    
    def _group_by_priority(self, nodes: List[SemanticNode], text: str) -> List[NodeGroup]:
        """按优先级引导的智能分组"""
        if not nodes:
            return []

        # 第一步：识别高优先级核心节点
        core_nodes = self._identify_core_nodes(nodes)

        # 第二步：为核心节点创建初始组
        groups = []
        core_groups = {}  # 核心节点到组的映射

        for core_node in core_nodes:
            group = NodeGroup()
            group.add_node(core_node)
            groups.append(group)
            core_groups[core_node] = group

        # 第三步：将其他节点分配到合适的组
        for node in nodes:
            if node in core_groups:
                continue  # 核心节点已经处理过

            # 查找最佳的目标组
            target_group = self._find_best_group(node, groups, text)

            if target_group and self._can_add_to_group_safely(target_group, node):
                target_group.add_node(node)
            else:
                # 创建新的独立组
                new_group = NodeGroup()
                new_group.add_node(node)
                groups.append(new_group)

        # 第四步：优化组大小和合并小组
        optimized_groups = self._optimize_groups(groups, text)

        # 第五步：最终处理
        for group in optimized_groups:
            self._finalize_group(group, text)

        return optimized_groups

    def _identify_core_nodes(self, nodes: List[SemanticNode]) -> List[SemanticNode]:
        """识别高优先级核心节点"""
        core_nodes = []

        for node in nodes:
            # 高优先级节点作为核心
            if node.priority >= self.config.high_priority_threshold:
                core_nodes.append(node)
            # 大型节点也可以作为核心（即使优先级不高）
            elif node.size > self.config.chunk_size * 0.4:
                core_nodes.append(node)

        return core_nodes

    def _find_best_group(self, node: SemanticNode, groups: List[NodeGroup], text: str) -> Optional[NodeGroup]:
        """为节点找到最佳的目标组"""
        best_group = None
        best_score = -1

        for group in groups:
            score = self._calculate_affinity_score(node, group, text)
            if score > best_score and score > 0:
                best_score = score
                best_group = group

        return best_group

    def _calculate_affinity_score(self, node: SemanticNode, group: NodeGroup, text: str) -> float:
        """计算节点与组的亲和度分数"""
        if not group.nodes:
            return 0

        score = 0

        # 1. 关系分数：如果节点与组内节点有关系
        for group_node in group.nodes:
            if group_node in node.related_nodes:
                score += 10  # 直接关系得高分

        # 2. 距离分数：物理距离越近分数越高
        min_distance = float('inf')
        for group_node in group.nodes:
            distance = abs(node.span.start - group_node.span.end)
            min_distance = min(min_distance, distance)

        # 距离分数（距离越近分数越高）
        if min_distance < 1000:  # 1000字节内
            score += (1000 - min_distance) / 100

        # 3. 优先级兼容性分数
        primary_node = group.primary_node
        if primary_node:
            # 高优先级节点可以吸引低优先级节点
            if primary_node.priority >= node.priority:
                score += (primary_node.priority - node.priority) * 0.5
            # 相同优先级节点容易合并
            elif primary_node.priority == node.priority:
                score += 2

        # 4. 类别兼容性分数
        if primary_node and primary_node.category == node.category:
            score += 3

        # 5. 大小限制：如果加入后超过限制，大幅降低分数
        if group.size + node.size > self.config.chunk_size:
            score -= 20

        return score
    
    def _can_add_to_group_safely(self, group: NodeGroup, node: SemanticNode) -> bool:
        """安全地检查是否可以将节点添加到组"""
        # 大小限制
        if group.size + node.size > self.config.chunk_size:
            return False

        # 如果组为空，可以添加
        if not group.nodes:
            return True

        # 检查兼容性
        return self._is_compatible_with_group(group, node)

    def _is_compatible_with_group(self, group: NodeGroup, node: SemanticNode) -> bool:
        """检查节点与组的兼容性"""
        primary_node = group.primary_node
        if not primary_node:
            return True

        # 1. 直接关系：如果有直接关系，高度兼容
        if primary_node in node.related_nodes or node in primary_node.related_nodes:
            return True

        # 2. 优先级兼容：高优先级可以吸引低优先级
        if primary_node.priority >= node.priority:
            return True

        # 3. 相同类别的节点兼容
        if primary_node.category == node.category:
            return True

        # 4. 特殊兼容规则
        return self._check_special_compatibility(primary_node, node)

    def _check_special_compatibility(self, primary_node: SemanticNode, node: SemanticNode) -> bool:
        """检查特殊的兼容性规则"""
        # 文档注释与代码元素兼容
        if (node.node_type.category == "documentation" and
            primary_node.category in ["structural", "functional"]):
            return True

        # 组织性节点（导入、导出）可以与任何节点兼容
        if node.node_type.category == "organizational":
            return True

        # 声明性节点可以与结构性节点兼容
        if (node.node_type.category == "declaration" and
            primary_node.category == "structural"):
            return True

        return False

    def _optimize_groups(self, groups: List[NodeGroup], text: str) -> List[NodeGroup]:
        """优化组：合并小组，拆分大组"""
        optimized = []

        for group in groups:
            if group.size < self.config.chunk_size * 0.1:  # 太小的组
                # 尝试合并到相邻的组
                merged = False
                for other_group in optimized:
                    if (other_group.size + group.size <= self.config.chunk_size and
                        self._can_merge_groups(other_group, group, text)):
                        # 合并组
                        for node in group.nodes:
                            other_group.add_node(node)
                        merged = True
                        break

                if not merged:
                    optimized.append(group)

            elif group.size > self.config.chunk_size * 1.5:  # 太大的组
                # 拆分大组
                split_groups = self._split_large_group(group, text)
                optimized.extend(split_groups)

            else:
                optimized.append(group)

        return optimized

    def _can_merge_groups(self, group1: NodeGroup, group2: NodeGroup, text: str) -> bool:
        """检查两个组是否可以合并"""
        if not group1.primary_node or not group2.primary_node:
            return True

        # 检查优先级兼容性
        if abs(group1.primary_node.priority - group2.primary_node.priority) <= 2:
            return True

        # 检查物理距离
        distance = abs(group1.span.end - group2.span.start) if group1.span and group2.span else 0
        if distance < 500:  # 500字节内
            return True

        return False

    def _split_large_group(self, group: NodeGroup, text: str) -> List[NodeGroup]:
        """拆分过大的组"""
        if len(group.nodes) <= 1:
            return [group]  # 无法拆分单节点组

        # 按优先级和位置重新分组
        sorted_nodes = sorted(group.nodes, key=lambda n: (n.span.start, -n.priority))

        split_groups = []
        current_group = NodeGroup()
        current_size = 0

        for node in sorted_nodes:
            if current_size + node.size > self.config.chunk_size and current_group.nodes:
                split_groups.append(current_group)
                current_group = NodeGroup()
                current_size = 0

            current_group.add_node(node)
            current_size += node.size

        if current_group.nodes:
            split_groups.append(current_group)

        return split_groups if split_groups else [group]
    
    def _add_related_nodes(self, group: NodeGroup, primary_node: SemanticNode, text: str):
        """添加与主节点相关的节点到组"""
        for related_node in primary_node.related_nodes:
            if related_node not in group.nodes:
                group.add_node(related_node)
    
    def _finalize_group(self, group: NodeGroup, text: str):
        """完成组的处理，包括行边界对齐"""
        if not group.span:
            return
        
        # 按行边界对齐
        aligned_span = self._align_to_line_boundaries(group.span, text)
        group.span = aligned_span
    
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
    
    def _create_documents(self, groups: List[NodeGroup], path: str, text: str) -> List[Document]:
        """从节点组生成最终文档"""
        documents = []
        
        for group in groups:
            if not group.span:
                continue
            
            content = text[group.span.start:group.span.end]
            
            # 生成文档元数据
            metadata = self._generate_document_metadata(group)
            
            document = Document(
                chunk_id=str(uuid.uuid4()),
                path=path,
                content=content,
                start_line=self._get_line_number(text, group.span.start),
                end_line=self._get_line_number(text, group.span.end)
            )
            
            # 可以将元数据存储在文档中（如果Document类支持）
            # document.metadata = metadata
            
            documents.append(document)
        
        return documents
    
    def _generate_document_metadata(self, group: NodeGroup) -> dict:
        """生成文档元数据"""
        metadata = {
            "node_count": len(group.nodes),
            "primary_node_type": group.primary_node.node_type.type_name if group.primary_node else None,
            "priority": group.priority,
            "categories": list(set(node.category for node in group.nodes)),
            "node_types": [node.node_type.type_name for node in group.nodes]
        }
        
        # 添加主要节点的名称
        if group.primary_node and group.primary_node.name:
            metadata["primary_name"] = group.primary_node.name
        
        return metadata
