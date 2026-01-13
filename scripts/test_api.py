"""Script to test the API endpoints"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_simple_query():
    """Test simple query"""
    print("\n=== Testing Simple Query ===")
    query = {"query": "What is artificial intelligence?"}
    
    response = requests.post(f"{BASE_URL}/ask", json=query)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['source']}")
        print(f"Reasoning: {result['reasoning']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_document_query():
    """Test query that should trigger document search"""
    print("\n=== Testing Document Query ===")
    query = {
        "query": "What is the company's leave policy?",
        "session_id": "test_session_1"
    }
    
    response = requests.post(f"{BASE_URL}/ask", json=query)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['source']}")
        print(f"Reasoning: {result['reasoning']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("AI Agent API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Simple Query", test_simple_query),
        ("Document Query", test_document_query)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' failed: {str(e)}")
            results[name] = False
        time.sleep(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    run_all_tests()