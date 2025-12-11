def solve_max_profit(n):
    """
    Solves the Max Profit problem using Dynamic Programming.
    
    Time: n units
    Buildings:
    - Theatre (T): Earnings $1500, Time 5, Area 2x1 (We assume infinite land so area doesn't limit us, only time)
    - Pub (P): Earnings $1000, Time 4, Area 1x1
    - Park (C): Earnings $3000, Time 10, Area 3x1 (PDF says 'Commercial Park earnings $3000' in table?) 
      WAIT: PDF text says "Commercial Park $2000" in one place?
      Let's re-read the extracted text carefully.
      
      Text snippet from PDF:
      "Commercial Park $2000" (line 4)
      But then it says "Test Case 1: Time 7 -> Earnings $3000 (T:1 ($1500?), P:0, C:0)? No wait."
      Let's check Test Case 1: Time 7. Output $3000.
      Solution: T:1, P:0, C:0. 
      If T takes 5 units and earns $1500. Total = $1500.
      Something is wrong.
      
      Let's look at Test Case 2: Time 8. Earnings $4500.
      Solution: T:1 P:0 C:0 ?? No way. 
      
      Let's re-read the PDF text provided in context.
      "Establishment Earnings Theatre $1500 Pub $1000 Commercial Park $2000" (Line 4)
      AND "Each unit of time that a building is operational, it earns him money."
      
      AH! "Each unit of time that a building is operational, it earns him money."
      This changes everything. It's not a one-time earnings.
      
      It implies:
      1. Build T (takes 5 units). From t=5 to t=n, it earns $1500 *per unit time*? 
      Or $1500 total?
      
      Re-reading: "Each unit of time that a building is operational, it earns him money."
      Establishment Earnings:
      Theatre $1500
      Pub $1000
      Commercial Park $2000 (I will assume per unit time?)
      
      Let's check Test Case 1: Time Unit: 7. Output: $3000.
      Solution: T:1, P:0, C:0.
      Logic: 
      Build T = 5 units.
      Remaining logic: Time 7.
      It is operational for (7 - 5) = 2 units.
      Earnings = 2 * $1500 = $3000. Matches!
      
      Test Case 2: Time Unit: 8. Output: $4500.
      Solution: T:1, P:0, C:0.
      Logic:
      Build T = 5 units.
      Operational for (8 - 5) = 3 units.
      Earnings = 3 * $1500 = $4500. Matches!
      
      Test Case 3: Time Unit: 13. Output: $16500.
      Solution: T:2, P:0, C:0.
      Logic:
      Strategy A: Build T1 (ends at 5). Operational for 13-5 = 8 units. Earns 8 * 1500 = 12000.
      Can we build another? "He cannot have two properties being developed in parallel".
      So at t=5, we start building T2?
      T2 takes 5 units. Ends at t=10.
      Operational for 13-10 = 3 units. Earns 3 * 1500 = 4500.
      Total = 12000 + 4500 = 16500. Matches!
      
      Wait, can we look at other options?
      Pub (P): Cost time 4. Earns 1000/unit.
      Park (C): Cost time 10. Earns 3000/unit? (PDF said $2000 in text but let's check).
       - If Park earns 2000.
         Time 13. Build C (10). Operational 3 units. 3 * 2000 = 6000. (Way less than 16500).
         
      So the Goal is to maximize:
      Sum of ( (Total_Time - finish_time_i) * Earning_Rate_i ) for all built properties.
      Constraint: Properties built sequentially. 
      Sum of build_times <= Total_Time.
    
    This is a variant of the Unbounded Knapsack Problem where value depends on placement order?
    Actually, since order matters (earlier built = more operational time), 
    we should always build the most efficient property FIRST?
    
    Let's check rates:
    T: 1500 / 5 (build time) -> No, earning is 1500 * (N - t_end).
    The "Value" of a building built at start time `current_time` is:
       (Total_Time - (current_time + build_duration)) * Rate
       
    Since (Total - t - duration) is linear, and we want to maximize sum,
    this looks like we just need strict ordering or just pure combinations?
    
    Actually, since (N - t_end) is larger for earlier buildings, we should put the building with HIGHEST EARNING RATE as early as possible?
    Rate T: 1500/unit.
    Rate P: 1000/unit.
    Rate C: 3000/unit? (Let's assume text was typo and C is higher, or just 2000).
    If C is 2000, T(1500) and P(1000).
    
    Let's check TC3 with Pubs.
    Time 13. 
    Build P1 (4). Ops 9. 9*1000 = 9000.
    Build P2 (8). Ops 5. 5*1000 = 5000.
    Build P3 (12). Ops 1. 1*1000 = 1000.
    Total = 15000. (Less than 16500).
    
    So T is better than P.
    
    What if C was $3000 (Maybe I misread "Commercial Park $2000" or there's a typo in my reading of the PDF extract)?
    The text summary I extracted said: "$2000".
    Let's assume $3000 for a moment to see if it makes sense.
    Time 13.
    Build C (10). Ops 3. 3*3000 = 9000. (Still bad).
    
    So T is the king for TC3.
    
    DP Approach:
    dp[t] = max earnings possible with t units of time available for *construction*.
    Actually, simply:
    f(time_remaining) = max profit.
    But profit depends on how long it runs.
    
    Standard Knapsack doesn't simplify perfectly because "Item Value" = Rate * (Remaining_Time_After_Build).
    Remaining_Time_After_Build is variable.
    
    Brute Force / Memoization DFS:
    def solve(time_left):
       if time_left <= 0: return 0, counts
       max_val = 0
       
       # Try building T
       if time_left > 5:
          remaining = time_left - 5
          val = remaining * 1500 + solve(remaining)
       
       # Try building P
       if time_left > 4: ...
    """
    
    rates = {
        'T': {'time': 5, 'earn': 1500},
        'P': {'time': 4, 'earn': 1000},
        'C': {'time': 10, 'earn': 3000} # Assuming 3000 to double check correctness or 2000? PDF said 2000. Let's stick to 2000 first, unless 3000 is needed. 
        # Actually TC1 output $3000. T=1. 1500*2 = 3000. Matches.
        # Wait, if C earns 2000. Time 10.
        # But let's look at the problem statement line 4 again. "Commercial Park $2000"
        # I'll stick to 2000.
    }
    
    memo = {}

    def dfs(remaining_time):
        if remaining_time in memo:
            return memo[remaining_time]
        
        best_profit = 0
        best_config = {'T': 0, 'P': 0, 'C': 0}
        
        # Options
        for type_code, props in rates.items():
            cost = props['time']
            rate = props['earn']
            
            if remaining_time > cost:
                # We build it.
                ops_time = remaining_time - cost
                current_profit = ops_time * rate
                
                # Recursive call
                future_profit, future_config = dfs(remaining_time - cost)
                
                total_profit = current_profit + future_profit
                
                if total_profit > best_profit:
                    best_profit = total_profit
                    best_config = future_config.copy()
                    best_config[type_code] += 1
        
        memo[remaining_time] = (best_profit, best_config)
        return best_profit, best_config

    return dfs(n)

if __name__ == "__main__":
    test_cases = [7, 8, 13]
    for t in test_cases:
        print(f"Time Unit: {t}")
        profit, config = solve_max_profit(t)
        print(f"Earnings: ${profit}")
        print(f"Solution: T: {config['T']} P: {config['P']} C: {config['C']}")
        print("-" * 20)
