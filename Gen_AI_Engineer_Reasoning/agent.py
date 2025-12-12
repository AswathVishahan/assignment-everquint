import os
import json
import time
import google.generativeai as genai
from prompts import PLANNER_PROMPT, EXECUTOR_PROMPT, VERIFIER_PROMPT

class ReasoningAgent:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def _call_llm(self, prompt, retries=3):
        for i in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e):
                    wait_time = (2 ** i) * 5  # 5s, 10s, 20s
                    print(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return f"Error calling LLM: {str(e)}"
        return "Error: Maximum retries (rate limit) exceeded."

    def solve(self, question):
        metadata = {"retries": 0, "logs": []}
        
        # 1. PLANNER
        plan_prompt = f"{PLANNER_PROMPT}\n\nUSER QUESTION: {question}\nPLAN:"
        plan = self._call_llm(plan_prompt)
        metadata["plan"] = plan
        metadata["logs"].append(f"PLAN generated:\n{plan}")

        # Retry Loop
        max_retries = 3
        current_feedback = ""
        
        for attempt in range(max_retries + 1):
            metadata["retries"] = attempt
            
            # 2. EXECUTOR
            exec_prompt = f"{EXECUTOR_PROMPT}\n\nQUESTION: {question}\nPLAN:\n{plan}"
            if current_feedback:
                exec_prompt += f"\n\nNOTE: Your previous attempt failed. Fix based on this feedback: {current_feedback}"
            
            execution_raw = self._call_llm(exec_prompt)
            
            # Extract Final Answer
            final_answer = "Unknown"
            if "FINAL_ANSWER:" in execution_raw:
                parts = execution_raw.split("FINAL_ANSWER:")
                reasoning_visible = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                reasoning_visible = execution_raw
                final_answer = execution_raw # Fallback
            
            metadata["logs"].append(f"EXECUTION (Attempt {attempt}):\n{execution_raw}")

            # 3. VERIFIER
            verify_prompt = f"{VERIFIER_PROMPT}\n\nQUESTION: {question}\nPROPOSED SOLUTION:\n{execution_raw}"
            verify_raw = self._call_llm(verify_prompt)
            
            # Parse JSON
            try:
                # Remove Markdown code blocks if present
                clean_json = verify_raw.replace("```json", "").replace("```", "").strip()
                verification = json.loads(clean_json)
            except json.JSONDecodeError:
                # If verifier outputs bad JSON, treat as fail
                verification = {"passed": False, "feedback": "Verifier output invalid JSON."}
            
            metadata["checks"] = metadata.get("checks", [])
            metadata["checks"].append({
                "check_name": f"Attempt {attempt} Verification",
                "passed": verification.get("passed", False),
                "details": verification.get("feedback", "No feedback")
            })

            if verification.get("passed", False):
                return {
                    "answer": final_answer,
                    "status": "success",
                    "reasoning_visible_to_user": reasoning_visible,
                    "metadata": metadata
                }
            else:
                current_feedback = verification.get("feedback", "")
                # Continue loop to correct
        
        # If exhausted
        return {
            "answer": final_answer,
            "status": "failed",
            "reasoning_visible_to_user": "Maximum retries exceeded. Last attempt provided.",
            "metadata": metadata
        }

if __name__ == "__main__":
    # Simple CLI Test
    agent = ReasoningAgent()
    print("Agent initialized. Type 'exit' to quit.")
    while True:
        q = input("\nEnter Question: ")
        if q.lower() == 'exit': break
        result = agent.solve(q)
        print(json.dumps(result, indent=2))
