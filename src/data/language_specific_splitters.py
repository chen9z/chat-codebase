"""
语言特定的分片器
为每种编程语言提供专门的处理逻辑
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum

from .splitter import (
    BaseSplitter, 
    SplitterConfig, 
    SemanticChunk, 
    Span, 
    Document,
    get_parser_for_language
)


class LanguageSpecificSemanticType(Enum):
    """语言特定的语义类型基类"""
    pass


# Python 特定的语义类型
class PythonSemanticType(LanguageSpecificSemanticType):
    CLASS = ("class", 10)
    FUNCTION = ("function", 8)
    ASYNC_FUNCTION = ("async_function", 8)
    METHOD = ("method", 8)
    PROPERTY = ("property", 6)
    DECORATOR = ("decorator", 5)
    IMPORT = ("import", 2)
    DOCSTRING = ("docstring", 7)
    EXCEPTION_HANDLER = ("exception_handler", 6)
    
    def __init__(self, type_name, priority):
        self.type_name = type_name
        self.priority = priority


# Java 特定的语义类型
class JavaSemanticType(LanguageSpecificSemanticType):
    CLASS = ("class", 10)
    INTERFACE = ("interface", 10)
    ENUM = ("enum", 9)
    METHOD = ("method", 8)
    CONSTRUCTOR = ("constructor", 8)
    FIELD = ("field", 5)
    ANNOTATION = ("annotation", 4)
    PACKAGE = ("package", 3)
    IMPORT = ("import", 2)
    INNER_CLASS = ("inner_class", 9)
    
    def __init__(self, type_name, priority):
        self.type_name = type_name
        self.priority = priority


# JavaScript/TypeScript 特定的语义类型
class JavaScriptSemanticType(LanguageSpecificSemanticType):
    CLASS = ("class", 10)
    INTERFACE = ("interface", 10)
    FUNCTION = ("function", 8)
    ARROW_FUNCTION = ("arrow_function", 8)
    METHOD = ("method", 8)
    CONSTRUCTOR = ("constructor", 8)
    PROPERTY = ("property", 6)
    TYPE_ALIAS = ("type_alias", 7)
    ENUM = ("enum", 9)
    IMPORT = ("import", 2)
    EXPORT = ("export", 3)
    
    def __init__(self, type_name, priority):
        self.type_name = type_name
        self.priority = priority


# Go 特定的语义类型
class GoSemanticType(LanguageSpecificSemanticType):
    FUNCTION = ("function", 8)
    METHOD = ("method", 8)
    TYPE = ("type", 9)
    STRUCT = ("struct", 10)
    INTERFACE = ("interface", 10)
    CONST = ("const", 4)
    VAR = ("var", 3)
    IMPORT = ("import", 2)
    PACKAGE = ("package", 1)
    
    def __init__(self, type_name, priority):
        self.type_name = type_name
        self.priority = priority


# Rust 特定的语义类型
class RustSemanticType(LanguageSpecificSemanticType):
    FUNCTION = ("function", 8)
    IMPL = ("impl", 9)
    STRUCT = ("struct", 10)
    ENUM = ("enum", 9)
    TRAIT = ("trait", 10)
    MODULE = ("module", 7)
    MACRO = ("macro", 6)
    USE = ("use", 2)
    CONST = ("const", 4)
    
    def __init__(self, type_name, priority):
        self.type_name = type_name
        self.priority = priority


# C/C++ 特定的语义类型
class CppSemanticType(LanguageSpecificSemanticType):
    CLASS = ("class", 10)
    STRUCT = ("struct", 10)
    FUNCTION = ("function", 8)
    NAMESPACE = ("namespace", 9)
    TEMPLATE = ("template", 8)
    ENUM = ("enum", 7)
    UNION = ("union", 8)
    TYPEDEF = ("typedef", 5)
    INCLUDE = ("include", 2)
    MACRO = ("macro", 4)
    
    def __init__(self, type_name, priority):
        self.type_name = type_name
        self.priority = priority


class LanguageSpecificSplitter(BaseSplitter, ABC):
    """语言特定分片器的基类"""
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        super().__init__(config)
        self.language = self.get_language_name()
    
    @abstractmethod
    def get_language_name(self) -> str:
        """返回语言名称"""
        pass
    
    @abstractmethod
    def get_semantic_type_enum(self):
        """返回该语言的语义类型枚举"""
        pass
    
    @abstractmethod
    def get_node_type_mapping(self) -> Dict[str, Any]:
        """返回AST节点类型到语义类型的映射"""
        pass
    
    @abstractmethod
    def calculate_language_specific_complexity(self, content: str) -> float:
        """计算语言特定的复杂度"""
        pass
    
    @abstractmethod
    def get_comment_patterns(self) -> Dict[str, str]:
        """返回该语言的注释模式"""
        pass
    
    @abstractmethod
    def should_merge_language_specific(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """语言特定的合并判断逻辑"""
        pass
    
    def split(self, path: str, text: str) -> List[Document]:
        """语言特定的分片逻辑"""
        # 使用缓存
        if self.config.enable_parsing_cache:
            from .splitter import get_file_hash, _content_hash_cache
            file_hash = get_file_hash(text)
            cache_key = f"lang_specific_{self.language}_{file_hash}_{self.config.chunk_size}"
            
            if cache_key in _content_hash_cache:
                cached_hash, cached_result = _content_hash_cache[cache_key]
                if cached_hash == file_hash:
                    return cached_result

        try:
            parser = get_parser_for_language(self.language)
            if parser is None:
                logging.warning(f"No parser available for language: {self.language}, falling back to default splitter")
                from .splitter import DefaultSplitter
                fallback_splitter = DefaultSplitter(config=self.config)
                return fallback_splitter.split(path, text)

            tree = parser.parse(bytes(text, 'utf-8'))
            root_node = tree.root_node

            if not root_node or root_node.type == "ERROR":
                logging.warning(f"Parse error for {path}, falling back to default splitter")
                from .splitter import DefaultSplitter
                fallback_splitter = DefaultSplitter(config=self.config)
                return fallback_splitter.split(path, text)

            # 阶段1: 解析所有语义块（使用语言特定逻辑）
            semantic_chunks = self._parse_language_specific_chunks(root_node, text)

            # 如果没有找到语义块，降级到默认分割器
            if not semantic_chunks:
                logging.info(f"No semantic chunks found for {path}, falling back to default splitter")
                from .splitter import DefaultSplitter
                fallback_splitter = DefaultSplitter(config=self.config)
                return fallback_splitter.split(path, text)

            # 阶段2: 使用语言特定的合并策略
            merged_spans = self._merge_by_language_specific_rules(semantic_chunks, text)

            # 生成最终文档
            result = self._create_language_specific_documents(merged_spans, path, text, semantic_chunks)

            # 缓存结果
            if self.config.enable_parsing_cache:
                from .splitter import _content_hash_cache, get_file_hash
                _content_hash_cache[cache_key] = (file_hash, result)

            return result

        except Exception as e:
            logging.error(f"Unexpected error parsing {path} with language-specific splitter: {e}, falling back to default splitter")
            from .splitter import DefaultSplitter
            fallback_splitter = DefaultSplitter(config=self.config)
            return fallback_splitter.split(path, text)
    
    def _parse_language_specific_chunks(self, root_node, text: str) -> List[SemanticChunk]:
        """解析语言特定的语义块"""
        chunks = []
        comments = self._collect_language_specific_comments(root_node)
        node_mapping = self.get_node_type_mapping()

        def traverse(node):
            if node.type in node_mapping:
                semantic_type = node_mapping[node.type]
                chunk_size = node.end_byte - node.start_byte
                
                # 使用语言特定的复杂度计算
                content = text[node.start_byte:node.end_byte]
                complexity_score = self.calculate_language_specific_complexity(content)

                chunk = SemanticChunk(
                    span=Span(node.start_byte, node.end_byte),
                    semantic_type=semantic_type,
                    node_type=node.type,
                    name=self._extract_language_specific_name(node, text),
                    size=chunk_size,
                    complexity_score=complexity_score
                )

                # 关联注释
                chunk.associated_comments = self._find_language_specific_comments(
                    node, comments, text
                )

                chunks.append(chunk)

            # 继续遍历子节点
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return sorted(chunks, key=lambda c: c.span.start)
    
    def _collect_language_specific_comments(self, root_node) -> List:
        """收集语言特定的注释"""
        comments = []
        comment_patterns = self.get_comment_patterns()

        def traverse(node):
            # 根据语言特定的注释模式识别注释
            if self._is_language_specific_comment(node.type):
                comments.append(node)
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return sorted(comments, key=lambda n: n.start_byte)
    
    def _is_language_specific_comment(self, node_type: str) -> bool:
        """判断是否是语言特定的注释节点"""
        # 可以在子类中重写以提供更精确的注释识别
        comment_types = {
            "python": ["comment"],
            "java": ["line_comment", "block_comment", "javadoc_comment"],
            "javascript": ["comment", "line_comment", "block_comment"],
            "typescript": ["comment", "line_comment", "block_comment"],
            "go": ["comment", "line_comment", "block_comment"],
            "rust": ["line_comment", "block_comment", "doc_comment"],
            "cpp": ["comment", "line_comment", "block_comment"],
            "c": ["comment", "line_comment", "block_comment"],
        }
        
        return node_type in comment_types.get(self.language, [])
    
    def _find_language_specific_comments(self, semantic_node, all_comments: List, text: str) -> List:
        """查找与语义节点关联的语言特定注释"""
        associated_comments = []
        semantic_start = semantic_node.start_byte

        for comment in all_comments:
            if comment.end_byte > semantic_start:
                break

            # 使用语言特定的注释关联逻辑
            if self._should_associate_comment(comment, semantic_node, text):
                associated_comments.append(comment)

        return associated_comments
    
    def _should_associate_comment(self, comment_node, semantic_node, text: str) -> bool:
        """判断注释是否应该与语义节点关联（可在子类中重写）"""
        semantic_start = semantic_node.start_byte
        comment_end = comment_node.end_byte
        
        # 基本距离检查
        if semantic_start - comment_end > self.config.max_comment_distance:
            return False
        
        # 行数检查
        between_text = text[comment_end:semantic_start]
        lines_between = between_text.count('\n')
        if lines_between > self.config.max_comment_lines_gap:
            return False
        
        return True
    
    def _extract_language_specific_name(self, node, text: str) -> str:
        """提取语言特定的节点名称（可在子类中重写）"""
        try:
            # 通用的名称提取逻辑
            for child in node.children:
                if child.type == "identifier":
                    return text[child.start_byte:child.end_byte]
            return node.type
        except:
            return "unknown"
    
    def _merge_by_language_specific_rules(self, semantic_chunks: List[SemanticChunk], text: str) -> List[Span]:
        """使用语言特定规则进行合并"""
        if not semantic_chunks:
            return []

        merged_spans = []
        current_group = []
        current_size = 0

        for chunk in semantic_chunks:
            chunk_size = chunk.size

            # 检查大小限制
            if chunk_size < self.config.min_semantic_chunk_size:
                current_group.append(chunk)
                current_size += chunk_size
                continue

            # 高优先级的大块独立成chunk
            if (chunk.semantic_type.priority >= 8 and
                chunk_size > self.config.chunk_size * self.config.high_priority_independence_ratio):

                # 先处理当前组
                if current_group:
                    merged_spans.extend(self._merge_language_specific_group(current_group))
                    current_group = []
                    current_size = 0

                # 高优先级块独立
                merged_spans.append(self._create_span_with_comments(chunk))

            # 可以合并的块
            elif current_size + chunk_size <= self.config.chunk_size:
                current_group.append(chunk)
                current_size += chunk_size

            # 当前组已满，开始新组
            else:
                if current_group:
                    merged_spans.extend(self._merge_language_specific_group(current_group))

                current_group = [chunk]
                current_size = chunk_size

        # 处理最后一组
        if current_group:
            merged_spans.extend(self._merge_language_specific_group(current_group))

        return merged_spans
    
    def _merge_language_specific_group(self, group: List[SemanticChunk]) -> List[Span]:
        """使用语言特定逻辑合并一组语义块"""
        if not group:
            return []

        # 按优先级和复杂度排序
        group.sort(key=lambda c: (c.semantic_type.priority, c.complexity_score), reverse=True)

        # 使用语言特定的合并判断
        if len(group) > 1:
            merged_groups = []
            current_subgroup = [group[0]]
            
            for i in range(1, len(group)):
                if self.should_merge_language_specific(current_subgroup[-1], group[i]):
                    current_subgroup.append(group[i])
                else:
                    merged_groups.append(current_subgroup)
                    current_subgroup = [group[i]]
            
            merged_groups.append(current_subgroup)
            
            # 为每个子组创建span
            spans = []
            for subgroup in merged_groups:
                start = min(c.span.start for c in subgroup)
                end = max(c.span.end for c in subgroup)
                
                # 包含所有相关注释
                all_comments = []
                for chunk in subgroup:
                    if chunk.associated_comments:
                        all_comments.extend(chunk.associated_comments)

                if all_comments:
                    comment_start = min(c.start_byte for c in all_comments)
                    start = min(start, comment_start)
                
                spans.append(Span(start, end))
            
            return spans
        
        # 单个块的情况
        return [self._create_span_with_comments(group[0])]
    
    def _create_span_with_comments(self, chunk: SemanticChunk) -> Span:
        """创建包含注释的span"""
        if not chunk.associated_comments:
            return chunk.span

        min_start = min(comment.start_byte for comment in chunk.associated_comments)
        min_start = min(min_start, chunk.span.start)
        max_end = max(chunk.span.end,
                     max(comment.end_byte for comment in chunk.associated_comments))

        return Span(min_start, max_end)
    
    def _create_language_specific_documents(self, spans: List[Span], path: str, text: str, semantic_chunks: List[SemanticChunk]) -> List[Document]:
        """生成包含语言特定信息的文档列表"""
        documents = []
        
        for span in spans:
            # 收集这个span包含的语义信息
            contained_chunks = [chunk for chunk in semantic_chunks 
                              if chunk.span.start >= span.start and chunk.span.end <= span.end]
            
            semantic_info = {
                "splitter": f"language_specific_{self.language}",
                "language": self.language,
                "chunk_types": [chunk.semantic_type.type_name for chunk in contained_chunks],
                "chunk_names": [chunk.name for chunk in contained_chunks if chunk.name],
                "complexity_scores": [chunk.complexity_score for chunk in contained_chunks],
                "has_comments": any(chunk.associated_comments for chunk in contained_chunks),
                "language_specific_features": self._extract_language_features(text[span.start:span.end])
            }
            
            documents.append(Document(
                chunk_id=self._generate_chunk_id(),
                path=path,
                content=text[span.start:span.end],
                start_line=self._get_line_number(text, span.start),
                end_line=self._get_line_number(text, span.end),
                semantic_info=semantic_info
            ))
        return documents
    
    def _extract_language_features(self, content: str) -> Dict[str, Any]:
        """提取语言特定的特征（可在子类中重写）"""
        return {}
    
    def _generate_chunk_id(self) -> str:
        """生成唯一的chunk ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _get_line_number(self, source_code: str, index: int) -> int:
        """获取字符索引对应的行号"""
        if index < 0:
            return 1

        if index >= len(source_code):
            return len(source_code.splitlines()) or 1

        total_chars = 0
        for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
            if total_chars + len(line) > index:
                return line_number
            total_chars += len(line)

        return len(source_code.splitlines()) or 1


class PythonSplitter(LanguageSpecificSplitter):
    """Python 特定的分片器"""
    
    def get_language_name(self) -> str:
        return "python"
    
    def get_semantic_type_enum(self):
        return PythonSemanticType
    
    def get_node_type_mapping(self) -> Dict[str, PythonSemanticType]:
        return {
            "class_definition": PythonSemanticType.CLASS,
            "function_definition": PythonSemanticType.FUNCTION,
            "async_function_definition": PythonSemanticType.ASYNC_FUNCTION,
            "decorated_definition": PythonSemanticType.DECORATOR,
            "import_statement": PythonSemanticType.IMPORT,
            "import_from_statement": PythonSemanticType.IMPORT,
        }
    
    def calculate_language_specific_complexity(self, content: str) -> float:
        """Python特定的复杂度计算"""
        score = 0.0
        
        # 基础长度分数
        score += len(content) / 1000.0
        
        # Python特定的复杂度指标
        python_keywords = [
            'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally',
            'with', 'def', 'class', 'lambda', 'yield', 'await', 'async'
        ]
        
        for keyword in python_keywords:
            count = content.count(f' {keyword} ') + content.count(f'\n{keyword} ')
            score += count * 0.3
        
        # 装饰器复杂度
        score += content.count('@') * 0.2
        
        # 列表推导式和生成器表达式
        score += content.count('[') * 0.1
        score += content.count('(') * 0.1
        
        # 异常处理复杂度
        score += content.count('except') * 0.4
        
        # 嵌套缩进级别（Python特有）
        lines = content.split('\n')
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)  # 假设4空格缩进
        score += max_indent * 0.5
        
        return score
    
    def get_comment_patterns(self) -> Dict[str, str]:
        return {
            "line_comment": "#",
            "docstring_single": '"""',
            "docstring_double": "'''",
        }
    
    def should_merge_language_specific(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """Python特定的合并逻辑"""
        # Python装饰器应该与被装饰的函数/类合并
        if (chunk1.semantic_type == PythonSemanticType.DECORATOR and 
            chunk2.semantic_type in [PythonSemanticType.FUNCTION, PythonSemanticType.CLASS]):
            return True
        
        # 导入语句可以合并
        if (chunk1.semantic_type == PythonSemanticType.IMPORT and 
            chunk2.semantic_type == PythonSemanticType.IMPORT):
            return True
        
        # docstring应该与函数/类合并
        if (chunk1.semantic_type == PythonSemanticType.DOCSTRING and 
            chunk2.semantic_type in [PythonSemanticType.FUNCTION, PythonSemanticType.CLASS]):
            return True
        
        return False
    
    def extract_python_specific_info(self, node, text: str) -> Dict:
        """提取Python特定的信息"""
        info = {}
        
        # 检测是否是异步函数
        if node.type == "async_function_definition":
            info["is_async"] = True
        
        # 检测装饰器
        content = text[node.start_byte:node.end_byte]
        if '@' in content:
            info["has_decorators"] = True
            info["decorator_count"] = content.count('@')
        
        # 检测docstring
        if '"""' in content or "'''" in content:
            info["has_docstring"] = True
        
        # 检测异常处理
        if 'try:' in content or 'except' in content:
            info["has_exception_handling"] = True
        
        return info
    
    def _extract_language_features(self, content: str) -> Dict[str, Any]:
        """提取Python特定的特征"""
        features = {}
        
        # Python特定的语法特征
        features["has_list_comprehension"] = '[' in content and 'for' in content and 'in' in content
        features["has_generator_expression"] = '(' in content and 'for' in content and 'in' in content
        features["has_lambda"] = 'lambda' in content
        features["has_decorators"] = '@' in content
        features["has_async_await"] = 'async' in content or 'await' in content
        features["has_context_manager"] = 'with' in content
        features["has_exception_handling"] = 'try:' in content or 'except' in content
        features["has_type_hints"] = '->' in content or ':' in content
        
        # 计算缩进级别（Python特有）
        lines = content.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent // 4)
        
        features["max_indent_level"] = max(indent_levels) if indent_levels else 0
        features["avg_indent_level"] = sum(indent_levels) / len(indent_levels) if indent_levels else 0
        
        # 检测特定的Python模式
        features["has_magic_methods"] = '__' in content and content.count('__') >= 2
        features["has_property_decorator"] = '@property' in content
        features["has_classmethod"] = '@classmethod' in content
        features["has_staticmethod"] = '@staticmethod' in content
        
        return features


class JavaSplitter(LanguageSpecificSplitter):
    """Java 特定的分片器"""
    
    def get_language_name(self) -> str:
        return "java"
    
    def get_semantic_type_enum(self):
        return JavaSemanticType
    
    def get_node_type_mapping(self) -> Dict[str, JavaSemanticType]:
        return {
            "class_declaration": JavaSemanticType.CLASS,
            "interface_declaration": JavaSemanticType.INTERFACE,
            "enum_declaration": JavaSemanticType.ENUM,
            "method_declaration": JavaSemanticType.METHOD,
            "constructor_declaration": JavaSemanticType.CONSTRUCTOR,
            "field_declaration": JavaSemanticType.FIELD,
            "annotation_type_declaration": JavaSemanticType.ANNOTATION,
            "package_declaration": JavaSemanticType.PACKAGE,
            "import_declaration": JavaSemanticType.IMPORT,
        }
    
    def calculate_language_specific_complexity(self, content: str) -> float:
        """Java特定的复杂度计算"""
        score = 0.0
        
        # 基础长度分数
        score += len(content) / 1000.0
        
        # Java特定的复杂度指标
        java_keywords = [
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch',
            'finally', 'synchronized', 'volatile', 'transient'
        ]
        
        for keyword in java_keywords:
            score += content.count(keyword) * 0.3
        
        # 泛型复杂度
        score += content.count('<') * 0.2
        score += content.count('>') * 0.2
        
        # 注解复杂度
        score += content.count('@') * 0.2
        
        # 异常处理复杂度
        score += content.count('throws') * 0.3
        score += content.count('catch') * 0.4
        
        # 继承和实现复杂度
        score += content.count('extends') * 0.3
        score += content.count('implements') * 0.3
        
        # 访问修饰符多样性
        modifiers = ['public', 'private', 'protected', 'static', 'final', 'abstract']
        modifier_count = sum(content.count(mod) for mod in modifiers)
        score += modifier_count * 0.1
        
        return score
    
    def get_comment_patterns(self) -> Dict[str, str]:
        return {
            "line_comment": "//",
            "block_comment": "/*",
            "javadoc": "/**",
        }
    
    def should_merge_language_specific(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """Java特定的合并逻辑"""
        # Javadoc应该与类/方法合并
        if (chunk1.name.startswith("/**") and 
            chunk2.semantic_type in [JavaSemanticType.CLASS, JavaSemanticType.METHOD]):
            return True
        
        # 导入语句可以合并
        if (chunk1.semantic_type == JavaSemanticType.IMPORT and 
            chunk2.semantic_type == JavaSemanticType.IMPORT):
            return True
        
        # 字段声明可以合并
        if (chunk1.semantic_type == JavaSemanticType.FIELD and 
            chunk2.semantic_type == JavaSemanticType.FIELD):
            return True
        
        # 内部类应该保持相对独立
        if chunk1.semantic_type == JavaSemanticType.INNER_CLASS:
            return False
        
        return False
    
    def _extract_language_features(self, content: str) -> Dict[str, Any]:
        """提取Java特定的特征"""
        features = {}
        
        # Java特定的语法特征
        features["has_generics"] = '<' in content and '>' in content
        features["has_annotations"] = '@' in content
        features["has_lambda"] = '->' in content
        features["has_streams"] = '.stream()' in content or '.parallelStream()' in content
        features["has_exception_handling"] = 'try' in content or 'catch' in content
        features["has_synchronized"] = 'synchronized' in content
        features["has_inheritance"] = 'extends' in content
        features["has_interface_impl"] = 'implements' in content
        features["has_static_imports"] = 'import static' in content
        
        # 访问修饰符统计
        features["has_public"] = 'public' in content
        features["has_private"] = 'private' in content
        features["has_protected"] = 'protected' in content
        features["has_static"] = 'static' in content
        features["has_final"] = 'final' in content
        features["has_abstract"] = 'abstract' in content
        
        # Java 8+ 特性
        features["has_optional"] = 'Optional' in content
        features["has_completable_future"] = 'CompletableFuture' in content
        
        # 设计模式检测
        features["possible_builder_pattern"] = 'Builder' in content and 'build()' in content
        features["possible_factory_pattern"] = 'Factory' in content
        features["possible_singleton_pattern"] = 'getInstance' in content
        
        # 框架检测
        features["spring_annotations"] = '@Autowired' in content or '@Component' in content or '@Service' in content
        features["junit_test"] = '@Test' in content
        
        return features


class JavaScriptSplitter(LanguageSpecificSplitter):
    """JavaScript/TypeScript 特定的分片器"""
    
    def get_language_name(self) -> str:
        return "javascript"
    
    def get_semantic_type_enum(self):
        return JavaScriptSemanticType
    
    def get_node_type_mapping(self) -> Dict[str, JavaScriptSemanticType]:
        return {
            "class_declaration": JavaScriptSemanticType.CLASS,
            "interface_declaration": JavaScriptSemanticType.INTERFACE,
            "function_declaration": JavaScriptSemanticType.FUNCTION,
            "arrow_function": JavaScriptSemanticType.ARROW_FUNCTION,
            "method_definition": JavaScriptSemanticType.METHOD,
            "function_expression": JavaScriptSemanticType.FUNCTION,
            "type_alias_declaration": JavaScriptSemanticType.TYPE_ALIAS,
            "enum_declaration": JavaScriptSemanticType.ENUM,
            "import_statement": JavaScriptSemanticType.IMPORT,
            "export_statement": JavaScriptSemanticType.EXPORT,
        }
    
    def calculate_language_specific_complexity(self, content: str) -> float:
        """JavaScript特定的复杂度计算"""
        score = 0.0
        
        # 基础长度分数
        score += len(content) / 1000.0
        
        # JavaScript特定的复杂度指标
        js_keywords = [
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch',
            'finally', 'async', 'await', 'yield', 'return'
        ]
        
        for keyword in js_keywords:
            score += content.count(keyword) * 0.3
        
        # 异步复杂度
        score += content.count('async') * 0.4
        score += content.count('await') * 0.4
        score += content.count('Promise') * 0.3
        
        # 箭头函数复杂度
        score += content.count('=>') * 0.2
        
        # 解构赋值复杂度
        score += content.count('{') * 0.1
        score += content.count('[') * 0.1
        
        # 类型注解复杂度（TypeScript）
        score += content.count(':') * 0.1
        score += content.count('<') * 0.15
        
        # 回调和高阶函数复杂度
        score += content.count('callback') * 0.3
        score += content.count('.map(') * 0.2
        score += content.count('.filter(') * 0.2
        score += content.count('.reduce(') * 0.3
        
        return score
    
    def get_comment_patterns(self) -> Dict[str, str]:
        return {
            "line_comment": "//",
            "block_comment": "/*",
            "jsdoc": "/**",
        }
    
    def should_merge_language_specific(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """JavaScript特定的合并逻辑"""
        # 导入语句可以合并
        if (chunk1.semantic_type == JavaScriptSemanticType.IMPORT and 
            chunk2.semantic_type == JavaScriptSemanticType.IMPORT):
            return True
        
        # 导出语句可以合并
        if (chunk1.semantic_type == JavaScriptSemanticType.EXPORT and 
            chunk2.semantic_type == JavaScriptSemanticType.EXPORT):
            return True
        
        # 类型定义可以合并
        if (chunk1.semantic_type == JavaScriptSemanticType.TYPE_ALIAS and 
            chunk2.semantic_type == JavaScriptSemanticType.TYPE_ALIAS):
            return True
        
        # 小的箭头函数可以合并
        if (chunk1.semantic_type == JavaScriptSemanticType.ARROW_FUNCTION and 
            chunk2.semantic_type == JavaScriptSemanticType.ARROW_FUNCTION and
            chunk1.size < 200 and chunk2.size < 200):
            return True
        
        return False
    
    def _extract_language_features(self, content: str) -> Dict[str, Any]:
        """提取JavaScript/TypeScript特定的特征"""
        features = {}
        
        # JavaScript特定的语法特征
        features["has_arrow_functions"] = '=>' in content
        features["has_async_await"] = 'async' in content or 'await' in content
        features["has_promises"] = 'Promise' in content or '.then(' in content
        features["has_destructuring"] = '{' in content and '}' in content and '=' in content
        features["has_template_literals"] = '`' in content
        features["has_spread_operator"] = '...' in content
        features["has_classes"] = 'class ' in content
        features["has_modules"] = 'import' in content or 'export' in content
        features["has_generators"] = 'function*' in content or 'yield' in content
        
        # 函数式编程特征
        features["has_map"] = '.map(' in content
        features["has_filter"] = '.filter(' in content
        features["has_reduce"] = '.reduce(' in content
        features["has_foreach"] = '.forEach(' in content
        features["has_higher_order_functions"] = 'callback' in content
        
        # TypeScript特定特征
        features["has_type_annotations"] = ':' in content and ('string' in content or 'number' in content or 'boolean' in content)
        features["has_interfaces"] = 'interface ' in content
        features["has_generics"] = '<T>' in content or '<K,' in content
        features["has_enums"] = 'enum ' in content
        features["has_type_aliases"] = 'type ' in content and '=' in content
        
        # 框架检测
        features["react_components"] = 'React' in content or 'useState' in content or 'useEffect' in content
        features["vue_components"] = 'Vue' in content or 'v-' in content
        features["angular_features"] = '@Component' in content or '@Injectable' in content
        features["node_modules"] = 'require(' in content
        
        # 测试框架
        features["jest_tests"] = 'describe(' in content or 'test(' in content or 'it(' in content
        features["mocha_tests"] = 'describe(' in content and 'it(' in content
        
        return features


class GoSplitter(LanguageSpecificSplitter):
    """Go 特定的分片器"""
    
    def get_language_name(self) -> str:
        return "go"
    
    def get_semantic_type_enum(self):
        return GoSemanticType
    
    def get_node_type_mapping(self) -> Dict[str, GoSemanticType]:
        return {
            "function_declaration": GoSemanticType.FUNCTION,
            "method_declaration": GoSemanticType.METHOD,
            "type_declaration": GoSemanticType.TYPE,
            "struct_type": GoSemanticType.STRUCT,
            "interface_type": GoSemanticType.INTERFACE,
            "const_declaration": GoSemanticType.CONST,
            "var_declaration": GoSemanticType.VAR,
            "import_declaration": GoSemanticType.IMPORT,
            "package_clause": GoSemanticType.PACKAGE,
        }
    
    def calculate_language_specific_complexity(self, content: str) -> float:
        """Go特定的复杂度计算"""
        score = 0.0
        
        # 基础长度分数
        score += len(content) / 1000.0
        
        # Go特定的复杂度指标
        go_keywords = [
            'if', 'else', 'for', 'switch', 'case', 'select', 'go', 'defer',
            'chan', 'range', 'type', 'interface', 'struct'
        ]
        
        for keyword in go_keywords:
            score += content.count(keyword) * 0.3
        
        # 并发复杂度
        score += content.count('go ') * 0.5  # goroutine
        score += content.count('chan') * 0.4  # channel
        score += content.count('select') * 0.5  # select statement
        score += content.count('defer') * 0.3  # defer statement
        
        # 错误处理复杂度（Go特有模式）
        score += content.count('if err != nil') * 0.3
        score += content.count('error') * 0.2
        
        # 接口复杂度
        score += content.count('interface{}') * 0.2
        
        # 指针操作复杂度
        score += content.count('*') * 0.1
        score += content.count('&') * 0.1
        
        return score
    
    def get_comment_patterns(self) -> Dict[str, str]:
        return {
            "line_comment": "//",
            "block_comment": "/*",
        }
    
    def should_merge_language_specific(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """Go特定的合并逻辑"""
        # 导入语句可以合并
        if (chunk1.semantic_type == GoSemanticType.IMPORT and 
            chunk2.semantic_type == GoSemanticType.IMPORT):
            return True
        
        # 常量声明可以合并
        if (chunk1.semantic_type == GoSemanticType.CONST and 
            chunk2.semantic_type == GoSemanticType.CONST):
            return True
        
        # 变量声明可以合并
        if (chunk1.semantic_type == GoSemanticType.VAR and 
            chunk2.semantic_type == GoSemanticType.VAR):
            return True
        
        # 类型定义相关的可以合并
        if (chunk1.semantic_type == GoSemanticType.TYPE and 
            chunk2.semantic_type == GoSemanticType.TYPE):
            return True
        
        return False
    
    def _extract_language_features(self, content: str) -> Dict[str, Any]:
        """提取Go特定的特征"""
        features = {}
        
        # Go特定的并发特征
        features["has_goroutines"] = 'go ' in content
        features["has_channels"] = 'chan' in content
        features["has_select"] = 'select' in content
        features["has_defer"] = 'defer' in content
        features["has_context"] = 'context.' in content
        features["has_waitgroup"] = 'WaitGroup' in content
        features["has_mutex"] = 'Mutex' in content or 'RWMutex' in content
        
        # 错误处理模式
        features["has_error_handling"] = 'if err != nil' in content
        features["error_returns"] = 'error' in content and 'return' in content
        features["has_panic_recover"] = 'panic(' in content or 'recover(' in content
        
        # 类型系统特征
        features["has_interfaces"] = 'interface{' in content
        features["has_structs"] = 'struct{' in content
        features["has_type_assertions"] = '.(' in content
        features["has_type_switches"] = 'type' in content and 'switch' in content
        features["has_pointers"] = '*' in content or '&' in content
        
        # 包和模块
        features["has_package_declaration"] = 'package ' in content
        features["has_imports"] = 'import' in content
        features["has_init_function"] = 'func init()' in content
        
        # 测试相关
        features["has_tests"] = 'func Test' in content or 'testing.' in content
        features["has_benchmarks"] = 'func Benchmark' in content
        
        # 现代Go特性
        features["has_generics"] = '[T' in content or '[K,' in content
        features["has_embed"] = 'embed.' in content
        features["has_build_tags"] = '// +build' in content
        
        return features


class RustSplitter(LanguageSpecificSplitter):
    """Rust 特定的分片器"""
    
    def get_language_name(self) -> str:
        return "rust"
    
    def get_semantic_type_enum(self):
        return RustSemanticType
    
    def get_node_type_mapping(self) -> Dict[str, RustSemanticType]:
        return {
            "function_item": RustSemanticType.FUNCTION,
            "impl_item": RustSemanticType.IMPL,
            "struct_item": RustSemanticType.STRUCT,
            "enum_item": RustSemanticType.ENUM,
            "trait_item": RustSemanticType.TRAIT,
            "mod_item": RustSemanticType.MODULE,
            "macro_definition": RustSemanticType.MACRO,
            "use_declaration": RustSemanticType.USE,
            "const_item": RustSemanticType.CONST,
        }
    
    def calculate_language_specific_complexity(self, content: str) -> float:
        """Rust特定的复杂度计算"""
        score = 0.0
        
        # 基础长度分数
        score += len(content) / 1000.0
        
        # Rust特定的复杂度指标
        rust_keywords = [
            'if', 'else', 'match', 'for', 'while', 'loop', 'impl', 'trait',
            'struct', 'enum', 'fn', 'macro', 'unsafe', 'async', 'await'
        ]
        
        for keyword in rust_keywords:
            score += content.count(keyword) * 0.3
        
        # 所有权和借用复杂度
        score += content.count('&') * 0.2  # 借用
        score += content.count('&mut') * 0.3  # 可变借用
        score += content.count('Box<') * 0.3  # 堆分配
        score += content.count('Rc<') * 0.4  # 引用计数
        score += content.count('Arc<') * 0.4  # 原子引用计数
        
        # 错误处理复杂度
        score += content.count('Result<') * 0.3
        score += content.count('Option<') * 0.2
        score += content.count('?') * 0.2  # 错误传播
        score += content.count('unwrap()') * 0.3
        
        # 模式匹配复杂度
        score += content.count('match') * 0.4
        score += content.count('=>') * 0.2
        
        # 生命周期复杂度
        score += content.count("'") * 0.2
        
        # 不安全代码复杂度
        score += content.count('unsafe') * 0.6
        
        # 泛型复杂度
        score += content.count('<T>') * 0.3
        score += content.count('where') * 0.3
        
        return score
    
    def get_comment_patterns(self) -> Dict[str, str]:
        return {
            "line_comment": "//",
            "block_comment": "/*",
            "doc_comment": "///",
            "inner_doc_comment": "//!",
        }
    
    def should_merge_language_specific(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """Rust特定的合并逻辑"""
        # use声明可以合并
        if (chunk1.semantic_type == RustSemanticType.USE and 
            chunk2.semantic_type == RustSemanticType.USE):
            return True
        
        # 常量声明可以合并
        if (chunk1.semantic_type == RustSemanticType.CONST and 
            chunk2.semantic_type == RustSemanticType.CONST):
            return True
        
        # impl块应该保持相对独立
        if chunk1.semantic_type == RustSemanticType.IMPL:
            return False
        
        # trait定义应该保持独立
        if chunk1.semantic_type == RustSemanticType.TRAIT:
            return False
        
        return False
    
    def _extract_language_features(self, content: str) -> Dict[str, Any]:
        """提取Rust特定的特征"""
        features = {}
        
        # 所有权和借用特征
        features["has_borrowing"] = '&' in content
        features["has_mutable_borrowing"] = '&mut' in content
        features["has_move_semantics"] = 'move' in content
        features["has_ownership_transfer"] = 'std::mem::' in content
        features["has_smart_pointers"] = 'Box<' in content or 'Rc<' in content or 'Arc<' in content
        features["has_reference_counting"] = 'Rc<' in content or 'Arc<' in content
        
        # 错误处理和选项类型
        features["has_result_type"] = 'Result<' in content
        features["has_option_type"] = 'Option<' in content
        features["has_error_propagation"] = '?' in content
        features["has_unwrap"] = 'unwrap()' in content or 'expect(' in content
        features["has_match_expressions"] = 'match ' in content
        
        # 并发和异步
        features["has_threads"] = 'thread::' in content
        features["has_async_await"] = 'async' in content or 'await' in content
        features["has_futures"] = 'Future' in content
        features["has_channels"] = 'mpsc::' in content or 'channel(' in content
        features["has_atomic"] = 'Atomic' in content
        features["has_arc_mutex"] = 'Arc<Mutex<' in content
        
        # 特征和泛型
        features["has_traits"] = 'trait ' in content
        features["has_impl_blocks"] = 'impl ' in content
        features["has_generics"] = '<T>' in content or '<T:' in content
        features["has_where_clauses"] = 'where ' in content
        features["has_associated_types"] = 'type ' in content and '=' in content
        
        # 内存安全特征
        features["has_unsafe_code"] = 'unsafe' in content
        features["has_raw_pointers"] = '*const' in content or '*mut' in content
        features["has_transmute"] = 'transmute' in content
        
        # 模式匹配
        features["has_pattern_matching"] = '=>' in content
        features["has_destructuring"] = 'let (' in content or 'let {' in content
        features["has_guards"] = 'if ' in content and '=>' in content
        
        # 宏系统
        features["has_macros"] = 'macro_rules!' in content or content.count('!') > 2
        features["has_derive_macros"] = '#[derive(' in content
        features["has_procedural_macros"] = 'proc_macro' in content
        
        # 生命周期
        features["has_lifetimes"] = "'" in content and ':' in content
        
        # 测试和文档
        features["has_tests"] = '#[test]' in content or '#[cfg(test)]' in content
        features["has_doc_tests"] = '///' in content and '```' in content
        features["has_benchmarks"] = '#[bench]' in content
        
        # 现代Rust特性
        features["has_const_generics"] = 'const ' in content and '<' in content
        features["has_async_traits"] = '#[async_trait]' in content
        
        return features


class CppSplitter(LanguageSpecificSplitter):
    """C/C++ 特定的分片器"""
    
    def get_language_name(self) -> str:
        return "cpp"
    
    def get_semantic_type_enum(self):
        return CppSemanticType
    
    def get_node_type_mapping(self) -> Dict[str, CppSemanticType]:
        return {
            "class_specifier": CppSemanticType.CLASS,
            "struct_specifier": CppSemanticType.STRUCT,
            "function_definition": CppSemanticType.FUNCTION,
            "namespace_definition": CppSemanticType.NAMESPACE,
            "template_declaration": CppSemanticType.TEMPLATE,
            "enum_specifier": CppSemanticType.ENUM,
            "union_specifier": CppSemanticType.UNION,
            "typedef_declaration": CppSemanticType.TYPEDEF,
            "preproc_include": CppSemanticType.INCLUDE,
            "preproc_def": CppSemanticType.MACRO,
        }
    
    def calculate_language_specific_complexity(self, content: str) -> float:
        """C/C++特定的复杂度计算"""
        score = 0.0
        
        # 基础长度分数
        score += len(content) / 1000.0
        
        # C/C++特定的复杂度指标
        cpp_keywords = [
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch',
            'class', 'struct', 'template', 'namespace', 'virtual', 'override'
        ]
        
        for keyword in cpp_keywords:
            score += content.count(keyword) * 0.3
        
        # 指针和引用复杂度
        score += content.count('*') * 0.2
        score += content.count('&') * 0.2
        score += content.count('->') * 0.3
        
        # 模板复杂度
        score += content.count('template<') * 0.5
        score += content.count('typename') * 0.3
        
        # 内存管理复杂度
        score += content.count('new') * 0.3
        score += content.count('delete') * 0.3
        score += content.count('malloc') * 0.4
        score += content.count('free') * 0.4
        
        # 继承和多态复杂度
        score += content.count('virtual') * 0.4
        score += content.count('override') * 0.3
        score += content.count('public:') * 0.2
        score += content.count('private:') * 0.2
        score += content.count('protected:') * 0.2
        
        # 预处理器复杂度
        score += content.count('#define') * 0.3
        score += content.count('#ifdef') * 0.3
        score += content.count('#ifndef') * 0.3
        
        return score
    
    def get_comment_patterns(self) -> Dict[str, str]:
        return {
            "line_comment": "//",
            "block_comment": "/*",
            "doxygen": "/**",
        }
    
    def should_merge_language_specific(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        """C/C++特定的合并逻辑"""
        # include语句可以合并
        if (chunk1.semantic_type == CppSemanticType.INCLUDE and 
            chunk2.semantic_type == CppSemanticType.INCLUDE):
            return True
        
        # 宏定义可以合并
        if (chunk1.semantic_type == CppSemanticType.MACRO and 
            chunk2.semantic_type == CppSemanticType.MACRO):
            return True
        
        # typedef声明可以合并
        if (chunk1.semantic_type == CppSemanticType.TYPEDEF and 
            chunk2.semantic_type == CppSemanticType.TYPEDEF):
            return True
        
        # 模板声明应该与其实现合并
        if (chunk1.semantic_type == CppSemanticType.TEMPLATE and 
            chunk2.semantic_type in [CppSemanticType.CLASS, CppSemanticType.FUNCTION]):
            return True
        
        return False
    
    def _extract_language_features(self, content: str) -> Dict[str, Any]:
        """提取C/C++特定的特征"""
        features = {}
        
        # C++特定的面向对象特征
        features["has_classes"] = 'class ' in content
        features["has_inheritance"] = 'public:' in content or 'private:' in content or 'protected:' in content
        features["has_virtual_functions"] = 'virtual' in content
        features["has_abstract_classes"] = 'virtual' in content and '= 0' in content
        features["has_operator_overloading"] = 'operator' in content
        features["has_constructors"] = content.count('(') > content.count('void')
        features["has_destructors"] = '~' in content
        
        # 模板和泛型编程
        features["has_templates"] = 'template<' in content
        features["has_template_specialization"] = 'template<>' in content
        features["has_typename"] = 'typename' in content
        features["has_template_metaprogramming"] = 'std::enable_if' in content or 'SFINAE' in content
        
        # 指针和内存管理
        features["has_pointers"] = '*' in content
        features["has_references"] = '&' in content and 'int&' in content
        features["has_smart_pointers"] = 'std::unique_ptr' in content or 'std::shared_ptr' in content
        features["has_manual_memory"] = 'new' in content or 'delete' in content
        features["has_malloc_free"] = 'malloc(' in content or 'free(' in content
        features["has_pointer_arithmetic"] = '++' in content and '*' in content
        
        # 现代C++特性
        features["has_auto_keyword"] = 'auto ' in content
        features["has_lambda_expressions"] = '[' in content and ']' in content and '(' in content
        features["has_range_based_for"] = 'for(' in content and ':' in content
        features["has_nullptr"] = 'nullptr' in content
        features["has_move_semantics"] = 'std::move' in content or '&&' in content
        features["has_constexpr"] = 'constexpr' in content
        features["has_static_assert"] = 'static_assert' in content
        
        # STL和标准库
        features["has_stl_containers"] = 'std::vector' in content or 'std::map' in content or 'std::set' in content
        features["has_stl_algorithms"] = 'std::sort' in content or 'std::find' in content
        features["has_iterators"] = 'iterator' in content or 'begin()' in content
        features["has_streams"] = 'std::cout' in content or 'std::cin' in content or 'iostream' in content
        
        # 并发编程
        features["has_threads"] = 'std::thread' in content
        features["has_mutex"] = 'std::mutex' in content or 'std::lock' in content
        features["has_atomic"] = 'std::atomic' in content
        features["has_futures"] = 'std::future' in content or 'std::async' in content
        
        # 异常处理
        features["has_exceptions"] = 'try' in content or 'catch' in content or 'throw' in content
        features["has_noexcept"] = 'noexcept' in content
        
        # 预处理器特征
        features["has_include_guards"] = '#ifndef' in content and '#define' in content and '#endif' in content
        features["has_pragma_once"] = '#pragma once' in content
        features["has_macros"] = '#define' in content
        features["has_conditional_compilation"] = '#ifdef' in content or '#ifndef' in content
        
        # C特定特征
        features["has_c_style_casts"] = '(' in content and ')' in content and '*' not in content
        features["has_function_pointers"] = '(*' in content and ')(' in content
        features["has_unions"] = 'union ' in content
        features["has_bit_fields"] = ':' in content and 'struct' in content
        
        # 框架和库检测
        features["has_boost"] = 'boost::' in content
        features["has_qt"] = 'QObject' in content or 'Q_OBJECT' in content
        features["has_opencv"] = 'cv::' in content
        
        return features


# 语言到分片器的映射
LANGUAGE_SPLITTER_MAPPING = {
    "python": PythonSplitter,
    "java": JavaSplitter,
    "javascript": JavaScriptSplitter,
    "typescript": JavaScriptSplitter,  # TypeScript使用相同的分片器
    "go": GoSplitter,
    "rust": RustSplitter,
    "cpp": CppSplitter,
    "c": CppSplitter,  # C使用相同的分片器
}


def get_language_specific_splitter(language: str, config: Optional[SplitterConfig] = None) -> LanguageSpecificSplitter:
    """根据语言获取对应的特定分片器"""
    splitter_class = LANGUAGE_SPLITTER_MAPPING.get(language.lower())
    
    if splitter_class:
        return splitter_class(config)
    else:
        # 如果没有特定的分片器，返回Python分片器作为默认值
        logging.warning(f"No specific splitter found for language: {language}, using Python splitter as fallback")
        return PythonSplitter(config)