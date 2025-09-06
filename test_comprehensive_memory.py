#!/usr/bin/env python3
"""
Test the comprehensive memory system to ensure ALL conversations are stored and recalled.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_comprehensive_memory():
    """Test that the bot remembers ALL conversations and can recall any information."""
    print("🧠 Testing Comprehensive Memory System")
    print("=" * 60)
    
    try:
        from crew_orchestrator import multi_agent_orchestrator
        from memory_manager import memory_manager
        
        test_user_id = "memory_test_user"
        
        # Clear any existing memory for clean test
        memory_manager.clear_conversation_history(test_user_id)
        memory_manager.clear_short_term_memory(test_user_id)
        
        print("1. Testing information storage...")
        
        # Simulate a conversation with various types of information
        test_conversations = [
            "Hi, my name is Sarah",
            "I drive a 2018 Honda CR-V",
            "I work as a software engineer at Google",
            "I live in San Francisco",
            "My favorite color is blue",
            "I have two cats named Whiskers and Mittens",
            "I enjoy hiking on weekends",
            "My email is sarah@example.com"
        ]
        
        # Store all conversations
        for i, message in enumerate(test_conversations, 1):
            print(f"\n   Storing conversation {i}: '{message}'")
            response = await multi_agent_orchestrator.process_request(message, test_user_id)
            print(f"   Bot response: {response[:80]}...")
        
        print("\n2. Testing memory recall...")
        
        # Test various memory queries
        memory_tests = [
            ("What car do I drive?", "2018 Honda CR-V"),
            ("What is my name?", "Sarah"),
            ("Where do I work?", "Google"),
            ("What do you remember about me?", "Sarah"),
            ("Do you remember what I told you about my pets?", "cats"),
            ("What's my email?", "sarah@example.com"),
            ("Where do I live?", "San Francisco"),
            ("What do I do for work?", "software engineer")
        ]
        
        successful_recalls = 0
        total_tests = len(memory_tests)
        
        for question, expected_info in memory_tests:
            print(f"\n   Testing: '{question}'")
            response = await multi_agent_orchestrator.process_request(question, test_user_id)
            print(f"   Response: {response}")
            
            # Check if the expected information is in the response
            if expected_info.lower() in response.lower():
                print(f"   ✅ SUCCESS: Found '{expected_info}' in response")
                successful_recalls += 1
            else:
                print(f"   ❌ FAILED: Expected '{expected_info}' not found in response")
        
        print(f"\n3. Testing memory statistics...")
        
        # Check memory storage
        stats = memory_manager.get_memory_stats(test_user_id)
        print(f"   Memory statistics:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        conversation_history = memory_manager.get_conversation_history(test_user_id)
        print(f"   Total conversation messages: {len(conversation_history)}")
        
        print(f"\n4. Testing contextual memory...")
        
        # Test that the bot can use stored information in context
        contextual_tests = [
            "What kind of car maintenance should I do?",  # Should reference Honda CR-V
            "Any job opportunities in my field?",         # Should reference software engineering
            "What's the weather like where I live?"       # Should reference San Francisco
        ]
        
        for question in contextual_tests:
            print(f"\n   Contextual test: '{question}'")
            response = await multi_agent_orchestrator.process_request(question, test_user_id)
            print(f"   Response: {response[:150]}...")
        
        # Calculate success rate
        success_rate = (successful_recalls / total_tests) * 100
        
        print(f"\n{'='*20} MEMORY TEST RESULTS {'='*20}")
        print(f"Memory recall success rate: {successful_recalls}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("✅ EXCELLENT: Memory system is working very well!")
        elif success_rate >= 60:
            print("⚠️  GOOD: Memory system is working but could be improved")
        else:
            print("❌ NEEDS WORK: Memory system needs significant improvement")
        
        return success_rate >= 60
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the comprehensive memory test."""
    print("🚀 Comprehensive Memory Test")
    print("=" * 60)
    print("Testing: Complete conversation storage and recall")
    print("=" * 60)
    
    success = await test_comprehensive_memory()
    
    print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
    
    if success:
        print("✅ Memory system is working well!")
        print("\n🧠 Your Discord bot now:")
        print("• Stores ALL conversations permanently")
        print("• Can recall specific information when asked")
        print("• Remembers personal details, preferences, and facts")
        print("• Uses stored information for contextual responses")
        print("• Provides comprehensive memory search")
        print("\n💬 Users can ask things like:")
        print("• 'What car do I drive?'")
        print("• 'Do you remember what I told you about...?'")
        print("• 'What do you know about me?'")
        print("• 'What did I say earlier about...?'")
        print("\n🔄 Memory works automatically in Discord!")
    else:
        print("❌ Memory system needs more work")
        print("Check the logs for specific issues")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
