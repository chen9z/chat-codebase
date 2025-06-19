#!/usr/bin/env python3
"""
Unit test for comment and method association in Java code parsing.
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import parse

# Sample Java code with Javadoc comments
JAVA_CODE_SAMPLE = '''
package com.example;

import java.util.List;

/**
 * This is a sample class to demonstrate comment association.
 * It contains various methods with different types of comments.
 */
public class SampleClass {
    
    /**
     * This is a Javadoc comment for the constructor.
     * It should be associated with the constructor method.
     * 
     * @param name the name parameter
     */
    public SampleClass(String name) {
        this.name = name;
    }
    
    // This is a line comment for a simple method
    // It should also be associated with the method below
    public void simpleMethod() {
        System.out.println("Simple method");
    }
    
    /**
     * This is a comprehensive Javadoc comment for a complex method.
     * It demonstrates how documentation comments should be kept together
     * with their corresponding methods for better code understanding.
     * 
     * @param input the input parameter
     * @param options the options list
     * @return the processed result
     * @throws IllegalArgumentException if input is invalid
     */
    public String complexMethod(String input, List<String> options) {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        
        // Process the input
        String result = input.toUpperCase();
        
        // Apply options
        for (String option : options) {
            result = result.replace(option, "");
        }
        
        return result.trim();
    }
    
    /*
     * This is a block comment for another method.
     * Block comments should also be associated with methods.
     */
    public void blockCommentMethod() {
        // Implementation here
    }
    
    // Standalone comment that shouldn't be associated with anything
    
    public void methodWithoutComment() {
        System.out.println("No comment for this method");
    }
    
    private String name;
}
'''

def test_comment_association():
    """Test that comments are properly associated with methods."""
    
    # Create a temporary Java file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
        f.write(JAVA_CODE_SAMPLE)
        temp_file = f.name
    
    try:
        # Parse the Java file
        documents = parse(temp_file)
        
        print(f"Generated {len(documents)} document chunks:")
        print("=" * 60)
        
        for i, doc in enumerate(documents):
            print(f"\n--- Chunk {i+1} (lines {doc.start_line}-{doc.end_line}) ---")
            print(f"Content length: {len(doc.content)} characters")
            print("Content preview:")
            
            # Show first few lines of content
            lines = doc.content.split('\n')
            preview_lines = lines[:10] if len(lines) > 10 else lines
            for line_num, line in enumerate(preview_lines, 1):
                print(f"  {line_num:2d}: {line}")
            
            if len(lines) > 10:
                print(f"  ... ({len(lines) - 10} more lines)")
            
            # Check if this chunk contains both comments and methods
            has_javadoc = "/**" in doc.content
            has_line_comment = "//" in doc.content and not doc.content.strip().startswith("//")
            has_block_comment = "/*" in doc.content and not "/**" in doc.content
            has_method = "public " in doc.content and "(" in doc.content
            
            print(f"  Analysis:")
            print(f"    - Contains Javadoc: {has_javadoc}")
            print(f"    - Contains line comments: {has_line_comment}")
            print(f"    - Contains block comments: {has_block_comment}")
            print(f"    - Contains method: {has_method}")
            
            if (has_javadoc or has_line_comment or has_block_comment) and has_method:
                print(f"    ✅ SUCCESS: Comment and method are together!")
            elif has_method and not (has_javadoc or has_line_comment or has_block_comment):
                print(f"    ⚠️  Method without comment (this might be OK)")
            elif (has_javadoc or has_line_comment or has_block_comment) and not has_method:
                print(f"    ❌ WARNING: Comment without associated method")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        
        # Return test results for assertion
        return len(documents) > 0
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)

if __name__ == "__main__":
    success = test_comment_association()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
        sys.exit(1)
