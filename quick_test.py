#!/usr/bin/env python3
"""Quick test for context7 MCP implementation in ollama_workflow.py"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ollama_workflow import review_node, WorkflowState

def test_context7_integration():
    """Test the context7 MCP integration in review_node"""
    
    # Create a test state with technical question
    test_state = WorkflowState({
        'processed_output': 'Reactでは、useState()フックを使って状態管理ができます。const [state, setState] = useState(initialValue)の形で使用します。',
        'original_user_input': 'ReactのuseStateフックの使い方を教えて',
        'messages': [],
        'iteration': 1,
        'user_input': '',
        'should_continue': False,
        'search_results': '',
        'recent_search_mode': False,
        'initial_output': '',
        'reviewed_output': '',
        'document_generated': False
    })
    
    print("🧪 Testing context7 MCP integration in review_node...")
    print("📝 Test input:")
    print(f"  Original question: {test_state['original_user_input']}")
    print(f"  Processed output: {test_state['processed_output']}")
    print()
    
    try:
        # Execute review_node with context7 MCP
        result_state = review_node(test_state)
        
        print("✅ Review completed successfully!")
        print("📋 Results:")
        print(f"  Reviewed output length: {len(result_state.get('reviewed_output', ''))}")
        
        reviewed_output = result_state.get('reviewed_output', '')
        if reviewed_output:
            print("📄 Reviewed output preview:")
            print("-" * 60)
            print(reviewed_output[:500] + '...' if len(reviewed_output) > 500 else reviewed_output)
            print("-" * 60)
            
            # Check if context7 was likely used (look for more detailed/corrected content)
            if len(reviewed_output) > len(test_state['processed_output']) * 1.5:
                print("✅ Context7 MCP likely provided enhanced review (content significantly expanded)")
            elif 'レビュー完了' in reviewed_output and '問題なし' not in reviewed_output:
                print("✅ Context7 MCP provided detailed review with corrections")
            else:
                print("ℹ️ Review completed but enhancement level unclear")
        else:
            print("❌ No reviewed output received")
            
        return bool(reviewed_output)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Context7 MCP Integration Test")
    print("=" * 60)
    
    success = test_context7_integration()
    
    print("\n" + "=" * 60)
    print(f"Test Result: {'✅ PASSED' if success else '❌ FAILED'}")
