#!/usr/bin/env python3
"""
测试语言特定分片器
验证每种编程语言的专门处理逻辑
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.language_specific_splitters import (
    get_language_specific_splitter,
    PythonSplitter,
    JavaSplitter,
    JavaScriptSplitter,
    GoSplitter,
    RustSplitter,
    CppSplitter,
    LANGUAGE_SPLITTER_MAPPING
)
from src.data.splitter import SplitterConfig

# 测试代码示例
TEST_CODES = {
    "python": '''
import asyncio
from typing import List, Optional

@dataclass
class User:
    """用户类"""
    name: str
    age: int
    
    @property
    def is_adult(self) -> bool:
        return self.age >= 18

async def fetch_users() -> List[User]:
    """异步获取用户列表"""
    try:
        # 模拟网络请求
        await asyncio.sleep(0.1)
        return [User("Alice", 25), User("Bob", 17)]
    except Exception as e:
        print(f"Error: {e}")
        return []

def process_users(users: List[User]) -> List[User]:
    """处理用户数据"""
    return [user for user in users if user.is_adult]
''',

    "java": '''
package com.example.service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * 用户服务类
 * 提供用户相关的业务逻辑
 */
@Service
public class UserService {
    
    private final UserRepository userRepository;
    
    @Autowired
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    
    /**
     * 获取所有成年用户
     * @return 成年用户列表
     */
    public List<User> getAdultUsers() {
        return userRepository.findAll()
            .stream()
            .filter(user -> user.getAge() >= 18)
            .collect(Collectors.toList());
    }
    
    @Transactional
    public Optional<User> createUser(String name, int age) throws ValidationException {
        if (name == null || name.trim().isEmpty()) {
            throw new ValidationException("Name cannot be empty");
        }
        
        User user = User.builder()
            .name(name)
            .age(age)
            .build();
            
        return Optional.of(userRepository.save(user));
    }
}
''',

    "javascript": '''
import React, { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * 用户管理组件
 */
const UserManager = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(false);
    
    // 获取用户数据
    const fetchUsers = async () => {
        setLoading(true);
        try {
            const response = await axios.get('/api/users');
            setUsers(response.data);
        } catch (error) {
            console.error('Failed to fetch users:', error);
        } finally {
            setLoading(false);
        }
    };
    
    useEffect(() => {
        fetchUsers();
    }, []);
    
    // 过滤成年用户
    const adultUsers = users.filter(user => user.age >= 18);
    
    // 创建新用户
    const createUser = async (userData) => {
        const response = await fetch('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData)
        });
        
        if (!response.ok) {
            throw new Error('Failed to create user');
        }
        
        return response.json();
    };
    
    return (
        <div className="user-manager">
            {loading ? <div>Loading...</div> : (
                <UserList users={adultUsers} onCreate={createUser} />
            )}
        </div>
    );
};

export default UserManager;
''',

    "go": '''
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// User 用户结构体
type User struct {
    ID   int    \`json:"id"\`
    Name string \`json:"name"\`
    Age  int    \`json:"age"\`
}

// UserService 用户服务接口
type UserService interface {
    GetUser(ctx context.Context, id int) (*User, error)
    CreateUser(ctx context.Context, user *User) error
}

// userServiceImpl 用户服务实现
type userServiceImpl struct {
    users map[int]*User
    mutex sync.RWMutex
}

// NewUserService 创建用户服务
func NewUserService() UserService {
    return &userServiceImpl{
        users: make(map[int]*User),
    }
}

// GetUser 获取用户
func (s *userServiceImpl) GetUser(ctx context.Context, id int) (*User, error) {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    
    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
        if user, exists := s.users[id]; exists {
            return user, nil
        }
        return nil, fmt.Errorf("user not found: %d", id)
    }
}

// CreateUser 创建用户
func (s *userServiceImpl) CreateUser(ctx context.Context, user *User) error {
    if user == nil {
        return fmt.Errorf("user cannot be nil")
    }
    
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    s.users[user.ID] = user
    return nil
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    service := NewUserService()
    
    // 创建用户
    go func() {
        defer func() {
            if r := recover(); r != nil {
                fmt.Printf("Recovered: %v\n", r)
            }
        }()
        
        user := &User{ID: 1, Name: "Alice", Age: 25}
        if err := service.CreateUser(ctx, user); err != nil {
            panic(err)
        }
    }()
    
    time.Sleep(time.Millisecond * 100)
    fmt.Println("User service started")
}
''',

    "rust": '''
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub age: u8,
}

#[derive(Debug, thiserror::Error)]
pub enum UserError {
    #[error("User not found: {id}")]
    NotFound { id: u64 },
    #[error("Validation failed: {message}")]
    ValidationError { message: String },
    #[error("Database error: {0}")]
    DatabaseError(String),
}

pub type Result<T> = std::result::Result<T, UserError>;

#[async_trait::async_trait]
pub trait UserRepository: Send + Sync {
    async fn find_by_id(&self, id: u64) -> Result<Option<User>>;
    async fn save(&self, user: User) -> Result<User>;
    async fn delete(&self, id: u64) -> Result<()>;
}

pub struct InMemoryUserRepository {
    users: Arc<RwLock<HashMap<u64, User>>>,
}

impl InMemoryUserRepository {
    pub fn new() -> Self {
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl UserRepository for InMemoryUserRepository {
    async fn find_by_id(&self, id: u64) -> Result<Option<User>> {
        let users = self.users.read().await;
        Ok(users.get(&id).cloned())
    }
    
    async fn save(&self, user: User) -> Result<User> {
        if user.name.trim().is_empty() {
            return Err(UserError::ValidationError {
                message: "Name cannot be empty".to_string(),
            });
        }
        
        let mut users = self.users.write().await;
        users.insert(user.id, user.clone());
        Ok(user)
    }
    
    async fn delete(&self, id: u64) -> Result<()> {
        let mut users = self.users.write().await;
        match users.remove(&id) {
            Some(_) => Ok(()),
            None => Err(UserError::NotFound { id }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_user_repository() -> Result<()> {
        let repo = InMemoryUserRepository::new();
        
        let user = User {
            id: 1,
            name: "Alice".to_string(),
            age: 25,
        };
        
        // 保存用户
        let saved_user = repo.save(user.clone()).await?;
        assert_eq!(saved_user.id, user.id);
        
        // 查找用户
        let found_user = repo.find_by_id(1).await?;
        assert!(found_user.is_some());
        assert_eq!(found_user.unwrap().name, "Alice");
        
        // 删除用户
        repo.delete(1).await?;
        let deleted_user = repo.find_by_id(1).await?;
        assert!(deleted_user.is_none());
        
        Ok(())
    }
}
''',

    "cpp": '''
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <future>
#include <algorithm>

namespace UserService {

/**
 * @brief 用户类
 */
class User {
private:
    int id_;
    std::string name_;
    int age_;
    
public:
    User(int id, const std::string& name, int age)
        : id_(id), name_(name), age_(age) {}
    
    // 移动构造函数
    User(User&& other) noexcept
        : id_(other.id_), name_(std::move(other.name_)), age_(other.age_) {}
    
    // 拷贝赋值运算符
    User& operator=(const User& other) {
        if (this != &other) {
            id_ = other.id_;
            name_ = other.name_;
            age_ = other.age_;
        }
        return *this;
    }
    
    // Getters
    int getId() const noexcept { return id_; }
    const std::string& getName() const noexcept { return name_; }
    int getAge() const noexcept { return age_; }
    
    // 判断是否成年
    bool isAdult() const noexcept { return age_ >= 18; }
};

/**
 * @brief 用户仓储接口
 */
template<typename T>
class IUserRepository {
public:
    virtual ~IUserRepository() = default;
    virtual std::future<std::shared_ptr<T>> findById(int id) = 0;
    virtual std::future<bool> save(const T& user) = 0;
    virtual std::future<std::vector<T>> findAll() = 0;
};

/**
 * @brief 内存用户仓储实现
 */
class InMemoryUserRepository : public IUserRepository<User> {
private:
    std::unordered_map<int, std::shared_ptr<User>> users_;
    mutable std::mutex mutex_;
    
public:
    std::future<std::shared_ptr<User>> findById(int id) override {
        return std::async(std::launch::async, [this, id]() -> std::shared_ptr<User> {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = users_.find(id);
            return (it != users_.end()) ? it->second : nullptr;
        });
    }
    
    std::future<bool> save(const User& user) override {
        return std::async(std::launch::async, [this, user]() -> bool {
            try {
                std::lock_guard<std::mutex> lock(mutex_);
                users_[user.getId()] = std::make_shared<User>(user);
                return true;
            } catch (const std::exception& e) {
                std::cerr << "Error saving user: " << e.what() << std::endl;
                return false;
            }
        });
    }
    
    std::future<std::vector<User>> findAll() override {
        return std::async(std::launch::async, [this]() -> std::vector<User> {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<User> result;
            result.reserve(users_.size());
            
            for (const auto& pair : users_) {
                result.emplace_back(*pair.second);
            }
            
            return result;
        });
    }
};

/**
 * @brief 用户服务类
 */
class UserService {
private:
    std::unique_ptr<IUserRepository<User>> repository_;
    
public:
    explicit UserService(std::unique_ptr<IUserRepository<User>> repo)
        : repository_(std::move(repo)) {}
    
    // 获取所有成年用户
    std::future<std::vector<User>> getAdultUsers() {
        return std::async(std::launch::async, [this]() -> std::vector<User> {
            auto allUsers = repository_->findAll().get();
            
            std::vector<User> adults;
            std::copy_if(allUsers.begin(), allUsers.end(), 
                        std::back_inserter(adults),
                        [](const User& user) { return user.isAdult(); });
            
            return adults;
        });
    }
    
    // 创建用户
    std::future<bool> createUser(int id, const std::string& name, int age) {
        return std::async(std::launch::async, [this, id, name, age]() -> bool {
            if (name.empty()) {
                return false;
            }
            
            User user(id, name, age);
            return repository_->save(user).get();
        });
    }
};

} // namespace UserService

int main() {
    using namespace UserService;
    
    auto repository = std::make_unique<InMemoryUserRepository>();
    UserService service(std::move(repository));
    
    // 创建用户
    auto createResult = service.createUser(1, "Alice", 25);
    if (createResult.get()) {
        std::cout << "User created successfully" << std::endl;
    }
    
    // 获取成年用户
    auto adultUsers = service.getAdultUsers().get();
    std::cout << "Found " << adultUsers.size() << " adult users" << std::endl;
    
    return 0;
}
'''
}

def test_language_specific_splitters():
    """测试所有语言特定的分片器"""
    print("🚀 测试语言特定分片器")
    print("=" * 60)
    
    config = SplitterConfig(
        chunk_size=1500,
        min_semantic_chunk_size=100,
        enable_parsing_cache=True
    )
    
    for lang, code in TEST_CODES.items():
        print(f"\n--- 测试 {lang.upper()} 分片器 ---")
        
        # 创建临时文件
        suffix_map = {
            "python": ".py",
            "java": ".java", 
            "javascript": ".js",
            "go": ".go",
            "rust": ".rs",
            "cpp": ".cpp"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix_map[lang], delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 获取语言特定分片器
            splitter = get_language_specific_splitter(lang, config)
            print(f"分片器类型: {type(splitter).__name__}")
            
            # 执行分片
            documents = splitter.split(temp_file, code)
            print(f"生成块数: {len(documents)}")
            
            # 分析语义信息
            for i, doc in enumerate(documents[:2]):  # 只显示前2个
                semantic_info = doc.semantic_info
                print(f"\n  块 {i+1} (行 {doc.start_line}-{doc.end_line}):")
                print(f"    语言: {semantic_info.get('language', 'unknown')}")
                print(f"    分片器: {semantic_info.get('splitter', 'unknown')}")
                print(f"    语义类型: {semantic_info.get('chunk_types', [])}")
                print(f"    代码块名称: {semantic_info.get('chunk_names', [])}")
                print(f"    复杂度评分: {[round(s, 2) for s in semantic_info.get('complexity_scores', [])]}")
                print(f"    包含注释: {semantic_info.get('has_comments', False)}")
                
                # 显示语言特定特征
                features = semantic_info.get('language_specific_features', {})
                if features:
                    feature_summary = []
                    for key, value in features.items():
                        if value:
                            feature_summary.append(key.replace('has_', '').replace('_', ' '))
                    
                    if feature_summary:
                        print(f"    语言特征: {', '.join(feature_summary[:5])}" + 
                              ("..." if len(feature_summary) > 5 else ""))
        
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        
        finally:
            os.unlink(temp_file)

def test_splitter_mapping():
    """测试分片器映射"""
    print("\n" + "=" * 60)
    print("测试分片器映射")
    print("=" * 60)
    
    for lang, splitter_class in LANGUAGE_SPLITTER_MAPPING.items():
        print(f"{lang}: {splitter_class.__name__}")
        
        # 测试实例化
        try:
            splitter = splitter_class()
            print(f"  ✅ {splitter.get_language_name()} 分片器创建成功")
        except Exception as e:
            print(f"  ❌ 创建失败: {e}")

def test_fallback_mechanism():
    """测试降级机制"""
    print("\n" + "=" * 60)
    print("测试降级机制")
    print("=" * 60)
    
    # 测试不支持的语言
    try:
        splitter = get_language_specific_splitter("unknown_language")
        print(f"✅ 不支持的语言降级成功: {type(splitter).__name__}")
    except Exception as e:
        print(f"❌ 降级失败: {e}")

def main():
    """运行所有测试"""
    try:
        test_language_specific_splitters()
        test_splitter_mapping()
        test_fallback_mechanism()
        
        print("\n" + "=" * 60)
        print("🎉 所有语言特定分片器测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()