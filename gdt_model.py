

# ---- DEFINITION OF MODEL PARAMETERS - Could be its own module/can be adapted per scenario  ----
# 1. Fixed parameters
W = 1.0     # Goal Value/Importance: Relative importance of a goal; goal_importance in Tomis model

BETA0 = 0.5    # positive Affect when goal state is reached or "baseline affect"; completion_affect in Tomis model

PI = 1      # Expectedness/Subjective Probability of a future/anticipated state; probability in Tomis Model

# 2. Salience components: The attention/activation given to each affect-generating component, i.e. completion, discrepancy, delta_discrepancy
S0 = 0.3    # Salience for goal completion/baseline; salience_c in Tomis model
S1 = 0.4    # Salience for discrepancy; salience_d in Tomis model
S3 = 0.3    # Salience for change in discrepancy; salience_delta_d in Tomis model

# 3. Additional parameters
    
MAX_DISCREPANCY = None  # Maximum discrepancy: a cap so that discrepancy cannot grow infintely



# ---- MODEL ----

# 1. Calculation of discrepancy

def discrepancy (state, goal, max_discrepancy=MAX_DISCREPANCY):
    """
    Calculates the sum of the feature differences as discrepancy, 
    for those features in the goal that are not set to None (those are ignored, this allows for multiple goals later)
    a = state feature; TODO rename it to s? But s is salience...
    b = goal feature; TODO rename it to g
    Remark: No homeostatic goals yet, only positive achievement goals
    """
    d = sum(abs(a - b) if b is not None else 0 for a, b in zip(state, goal))
    if max_discrepancy is not None:
        d = min(d, max_discrepancy)
    return d


# 2. Calculation of change in discrepancy

def delta_discrepancy (prev_d, current_d):
    delta_d = prev_d - current_d
    # TODO: Right now I dropped the normalization of this delta_d that we had noted down in the original version of the code. 
    # it was dropped because it seemed not to fit anymore with the code that we had written (in my understanding, please correct me)
    return delta_d
    


# 3. Calculation of the three types of Affect + a combined Affect

# a) Affect from goal completion (baseline affect)
def affect_from_completion(w, B0, s0):
    return s0 * w * B0

# b) Affect from Discrepancy
def affect_discrepancy(w, B0, d, s1):
    return s1 * w * (B0 + d)

# c) Affect from Change in Discrepancy
def affect_delta_discrepancy(w, B0, delta_d, s2):
    return -s2 * w * (B0 + delta_d)

# d) Calculation of Combined Affect
def combined_affect(w, B0, d, delta_d, s0, s1, s2, pi=None):
    """
    Combines the three affect components and optionally applies subjective probability (pi).
    """
    a0 = affect_from_completion(w, B0, s0)
    a_discr = affect_discrepancy(w, B0, d, s1)
    a_deltadiscr = affect_delta_discrepancy(w, B0, delta_d, s2)

    a_total = a0 - a_discr - a_deltadiscr

    if pi is not None:
        return a_total * pi
    return a_total





