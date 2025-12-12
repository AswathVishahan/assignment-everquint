import json
import time
from agent import ReasoningAgent

def run_tests():
    agent = ReasoningAgent()
    
    test_cases = [
        # Easy
        "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
        "What is 25 * 4 + 10?",
        "Alice has 3 red apples and twice as many green apples. How many total?",
        "If I run 5km in 30 minutes, what is my average speed in km/h?",
        "A rectangular garden is 10m by 5m. What is the area?",
        
        # Tricky / Multi-step
        "I have a 12 liter jug and a 5 liter jug. How exactly do I measure 4 liters?",
        "A meeting needs 60 minutes. Free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. Which can fit it?",
        "John is 3 times as old as Mary. In 10 years, he will be twice as old as her. How old is Mary now?"
    ]

    print(f"Running {len(test_cases)} tests...\n")

    results = []
    
    for i, q in enumerate(test_cases):
        print(f"Test {i+1}: {q}")
        try:
            res = agent.solve(q)
            status = res['status']
            ans = res['answer']
            print(f"  -> Status: {status}")
            print(f"  -> Answer: {ans}")
            print(f"  -> Retries: {res['metadata']['retries']}")
            results.append(res)
        except Exception as e:
            print(f"  -> STARTED ERROR: {e}")

        print("-" * 40)
        time.sleep(10) # Heavy wait to start next test

    # Save logs
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull logs saved to test_results.json")

if __name__ == "__main__":
    run_tests()
