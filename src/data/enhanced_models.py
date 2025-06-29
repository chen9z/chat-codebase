"""
增强的语义分析数据模型
支持通用节点解析和优先级合并
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


class NodeType(Enum):
    """节点类型枚举，包含所有可能的代码元素"""
    
    # 高优先级结构节点
    CLASS = ("class", 10, "structural")
    INTERFACE = ("interface", 10, "structural") 
    ENUM = ("enum", 9, "structural")
    NAMESPACE = ("namespace", 9, "structural")
    MODULE = ("module", 9, "structural")
    
    # 高优先级功能节点
    FUNCTION = ("function", 8, "functional")
    METHOD = ("method", 8, "functional")
    CONSTRUCTOR = ("constructor", 7, "functional")
    DESTRUCTOR = ("destructor", 7, "functional")
    
    # 中优先级声明节点
    PROPERTY = ("property", 6, "declaration")
    FIELD = ("field", 5, "declaration")
    VARIABLE = ("variable", 4, "declaration")
    CONSTANT = ("constant", 5, "declaration")
    TYPE_ALIAS = ("type_alias", 6, "declaration")
    
    # 文档和注释节点
    DOC_COMMENT = ("doc_comment", 7, "documentation")  # 文档注释优先级较高
    BLOCK_COMMENT = ("block_comment", 3, "documentation")
    LINE_COMMENT = ("line_comment", 2, "documentation")
    
    # 低优先级组织节点
    IMPORT = ("import", 2, "organizational")
    EXPORT = ("export", 2, "organizational")
    PACKAGE = ("package", 1, "organizational")
    
    # 控制流节点
    IF_STATEMENT = ("if_statement", 3, "control_flow")
    LOOP = ("loop", 3, "control_flow")
    TRY_CATCH = ("try_catch", 4, "control_flow")
    
    # 其他节点
    EXPRESSION = ("expression", 1, "other")
    STATEMENT = ("statement", 1, "other")
    OTHER = ("other", 0, "other")
    
    def __init__(self, type_name: str, priority: int, category: str):
        self.type_name = type_name
        self.priority = priority
        self.category = category


@dataclass
class SemanticNode:
    """通用语义节点，表示代码中的任何有意义的元素"""

    # 基本信息
    node_type: NodeType
    ast_node_type: str  # 原始AST节点类型
    span: 'Span'
    name: str = ""

    # 内容信息
    content: str = ""

    # 关系信息
    parent: Optional['SemanticNode'] = field(default=None, compare=False)
    children: List['SemanticNode'] = field(default_factory=list, compare=False)
    related_nodes: List['SemanticNode'] = field(default_factory=list, compare=False)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __hash__(self):
        """使节点可以用作字典键或集合元素"""
        return hash((self.node_type, self.ast_node_type, self.span.start, self.span.end, self.name))

    def __eq__(self, other):
        """节点相等性比较"""
        if not isinstance(other, SemanticNode):
            return False
        return (self.node_type == other.node_type and
                self.ast_node_type == other.ast_node_type and
                self.span.start == other.span.start and
                self.span.end == other.span.end and
                self.name == other.name)
    
    @property
    def size(self) -> int:
        """节点大小（字节数）"""
        return self.span.size
    
    @property
    def priority(self) -> int:
        """节点优先级"""
        return self.node_type.priority
    
    @property
    def category(self) -> str:
        """节点类别"""
        return self.node_type.category
    
    def add_child(self, child: 'SemanticNode'):
        """添加子节点"""
        child.parent = self
        self.children.append(child)
    
    def add_related(self, node: 'SemanticNode'):
        """添加相关节点（如注释关联到函数）"""
        self.related_nodes.append(node)


@dataclass
class Span:
    """表示文本中的一个范围"""
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def __hash__(self):
        """使Span可以用作字典键"""
        return hash((self.start, self.end))


class NodeClassifier(ABC):
    """节点分类器抽象基类"""
    
    @abstractmethod
    def classify(self, ast_node, language: str) -> NodeType:
        """将AST节点分类为语义节点类型"""
        pass
    
    @abstractmethod
    def extract_metadata(self, ast_node, text: str) -> Dict[str, Any]:
        """提取节点元数据"""
        pass


class DefaultNodeClassifier(NodeClassifier):
    """默认节点分类器"""
    
    # AST节点类型到语义节点类型的映射
    NODE_TYPE_MAPPING = {
        # Python
        "class_definition": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "async_function_definition": NodeType.FUNCTION,
        "decorated_definition": NodeType.FUNCTION,
        "import_statement": NodeType.IMPORT,
        "import_from_statement": NodeType.IMPORT,
        "assignment": NodeType.VARIABLE,
        "comment": NodeType.LINE_COMMENT,
        
        # Java
        "class_declaration": NodeType.CLASS,
        "interface_declaration": NodeType.INTERFACE,
        "method_declaration": NodeType.METHOD,
        "constructor_declaration": NodeType.CONSTRUCTOR,
        "enum_declaration": NodeType.ENUM,
        "field_declaration": NodeType.FIELD,
        "import_declaration": NodeType.IMPORT,
        "line_comment": NodeType.LINE_COMMENT,
        "block_comment": NodeType.BLOCK_COMMENT,
        "javadoc_comment": NodeType.DOC_COMMENT,
        
        # JavaScript/TypeScript
        "class_declaration": NodeType.CLASS,
        "function_declaration": NodeType.FUNCTION,
        "method_definition": NodeType.METHOD,
        "arrow_function": NodeType.FUNCTION,
        "function_expression": NodeType.FUNCTION,
        "variable_declaration": NodeType.VARIABLE,
        "import_statement": NodeType.IMPORT,
        "export_statement": NodeType.EXPORT,
        
        # Go
        "function_declaration": NodeType.FUNCTION,
        "method_declaration": NodeType.METHOD,
        "type_declaration": NodeType.CLASS,
        "interface_type": NodeType.INTERFACE,
        "struct_type": NodeType.CLASS,
        "import_spec": NodeType.IMPORT,
        "package_clause": NodeType.PACKAGE,
        
        # Rust
        "function_item": NodeType.FUNCTION,
        "impl_item": NodeType.CLASS,
        "struct_item": NodeType.CLASS,
        "enum_item": NodeType.ENUM,
        "trait_item": NodeType.INTERFACE,
        "mod_item": NodeType.MODULE,
        "use_declaration": NodeType.IMPORT,
        "doc_comment": NodeType.DOC_COMMENT,
        
        # C/C++
        "function_definition": NodeType.FUNCTION,
        "class_specifier": NodeType.CLASS,
        "struct_specifier": NodeType.CLASS,
        "namespace_definition": NodeType.NAMESPACE,
        "enum_specifier": NodeType.ENUM,
        "preproc_include": NodeType.IMPORT,
    }
    
    def classify(self, ast_node, language: str) -> NodeType:
        """将AST节点分类为语义节点类型"""
        node_type = ast_node.type
        
        # 首先尝试直接映射
        if node_type in self.NODE_TYPE_MAPPING:
            return self.NODE_TYPE_MAPPING[node_type]
        
        # 语言特定的分类逻辑
        return self._classify_by_language(ast_node, language)
    
    def _classify_by_language(self, ast_node, language: str) -> NodeType:
        """基于语言的特定分类逻辑"""
        node_type = ast_node.type
        
        # 通用模式匹配
        if "class" in node_type:
            return NodeType.CLASS
        elif "function" in node_type or "method" in node_type:
            return NodeType.FUNCTION
        elif "interface" in node_type:
            return NodeType.INTERFACE
        elif "enum" in node_type:
            return NodeType.ENUM
        elif "import" in node_type:
            return NodeType.IMPORT
        elif "comment" in node_type:
            if "doc" in node_type or "javadoc" in node_type:
                return NodeType.DOC_COMMENT
            elif "block" in node_type:
                return NodeType.BLOCK_COMMENT
            else:
                return NodeType.LINE_COMMENT
        else:
            return NodeType.OTHER
    
    def extract_metadata(self, ast_node, text: str) -> Dict[str, Any]:
        """提取节点元数据"""
        metadata = {
            "ast_type": ast_node.type,
            "start_line": self._get_line_number(text, ast_node.start_byte),
            "end_line": self._get_line_number(text, ast_node.end_byte),
        }
        
        # 提取名称
        name = self._extract_name(ast_node, text)
        if name:
            metadata["name"] = name
        
        # 提取修饰符
        modifiers = self._extract_modifiers(ast_node, text)
        if modifiers:
            metadata["modifiers"] = modifiers
        
        return metadata
    
    def _extract_name(self, ast_node, text: str) -> str:
        """提取节点名称"""
        # 查找identifier子节点
        for child in ast_node.children:
            if child.type == "identifier":
                return text[child.start_byte:child.end_byte]
        return ""
    
    def _extract_modifiers(self, ast_node, text: str) -> List[str]:
        """提取修饰符（如public, private, static等）"""
        modifiers = []
        for child in ast_node.children:
            if child.type in ["public", "private", "protected", "static", "final", "abstract"]:
                modifiers.append(child.type)
        return modifiers
    
    def _get_line_number(self, text: str, byte_index: int) -> int:
        """获取字节索引对应的行号"""
        return text[:byte_index].count('\n') + 1


@dataclass
class NodeGroup:
    """节点组，表示一组相关的语义节点"""
    
    nodes: List[SemanticNode] = field(default_factory=list)
    primary_node: Optional[SemanticNode] = None  # 主要节点
    span: Optional[Span] = None
    
    @property
    def priority(self) -> int:
        """组的优先级（取最高优先级节点）"""
        if not self.nodes:
            return 0
        return max(node.priority for node in self.nodes)
    
    @property
    def size(self) -> int:
        """组的大小"""
        if self.span:
            return self.span.size
        if not self.nodes:
            return 0
        return sum(node.size for node in self.nodes)
    
    def add_node(self, node: SemanticNode):
        """添加节点到组"""
        self.nodes.append(node)
        if not self.primary_node or node.priority > self.primary_node.priority:
            self.primary_node = node
        self._update_span()
    
    def _update_span(self):
        """更新组的span范围"""
        if not self.nodes:
            self.span = None
            return
        
        min_start = min(node.span.start for node in self.nodes)
        max_end = max(node.span.end for node in self.nodes)
        self.span = Span(min_start, max_end)


class RelationshipAnalyzer:
    """节点关系分析器"""
    
    def __init__(self, max_distance_lines: int = 3):
        self.max_distance_lines = max_distance_lines
    
    def analyze_relationships(self, nodes: List[SemanticNode], text: str) -> List[SemanticNode]:
        """分析节点之间的关系"""
        # 按位置排序
        sorted_nodes = sorted(nodes, key=lambda n: n.span.start)
        
        # 分析邻近关系
        for i, node in enumerate(sorted_nodes):
            # 查找前面的相关节点
            for j in range(i-1, -1, -1):
                prev_node = sorted_nodes[j]
                if self._are_related(prev_node, node, text):
                    node.add_related(prev_node)
                else:
                    break  # 距离太远，停止查找
        
        return sorted_nodes
    
    def _are_related(self, node1: SemanticNode, node2: SemanticNode, text: str) -> bool:
        """判断两个节点是否相关"""
        # 计算行距
        line1 = self._get_line_number(text, node1.span.end)
        line2 = self._get_line_number(text, node2.span.start)
        line_distance = line2 - line1
        
        if line_distance > self.max_distance_lines:
            return False
        
        # 特殊关系规则
        return self._check_special_relationships(node1, node2, text, line_distance)
    
    def _check_special_relationships(self, node1: SemanticNode, node2: SemanticNode, 
                                   text: str, line_distance: int) -> bool:
        """检查特殊关系"""
        # 文档注释与代码元素的关系
        if (node1.node_type == NodeType.DOC_COMMENT and 
            node2.node_type.category in ["structural", "functional"]):
            return line_distance <= 2
        
        # 普通注释与代码的关系
        if (node1.node_type in [NodeType.LINE_COMMENT, NodeType.BLOCK_COMMENT] and
            node2.node_type.category in ["structural", "functional"]):
            return line_distance <= 1
        
        # 导入语句的聚合
        if (node1.node_type == NodeType.IMPORT and node2.node_type == NodeType.IMPORT):
            return line_distance <= 5
        
        # 变量声明的聚合
        if (node1.node_type == NodeType.VARIABLE and node2.node_type == NodeType.VARIABLE):
            return line_distance <= 2
        
        return False
    
    def _get_line_number(self, text: str, byte_index: int) -> int:
        """获取字节索引对应的行号"""
        return text[:byte_index].count('\n') + 1
