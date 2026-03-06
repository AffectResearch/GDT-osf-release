class AffectModel:

    def __init__ (self, v=1.0, targets=None, expectancy=None, w1=None, w2=None):
        self.v = v  # Goal Importance
        self.targets = targets
        self.expectancy = expectancy
        self.utility_matrix = []
        self.w1 = w1     # representational strength of AD
        self.w2 = w2     # representational strength of AAS
        self.goal_list = []
        self.g = 1
        self.discrepancy_buffer = {}


    def discrepancy(self, perceived_features):
        for feature, target in self.targets.items():
            feature_val = perceived_features.get(feature, 0)
            #if feature > target:
            #    print(f"Warning: Feature ({feature}) > Target ({target})")
            d = target - feature_val
            #if self.max_discrepancy is not None:
            #    d = min(d, self.max_discrepancy)
            self.discrepancy_buffer[feature] = d
        return self.discrepancy_buffer
    
    def ad(self, feature_name):                          # Affect from Discrepancy Detection
        d = self.discrepancy_buffer[feature_name]
        ad = self.v if d==0 else self.v*d
        return ad
    
    def utilities(self):
        e = self.expectancy
        u = e*self.v
        utilities = self.utility_matrix.append(u)
        return u, utilities
    
    def aas(self):
        aas = max(self.utility_matrix)  # function can be changed
        return aas
    
    def aff_comp(self, perceived_features):
        self.discrepancy(perceived_features)
        ad_sum = 0
        for feature in self.targets.keys():
            ad = self.ad(feature)
            ad_sum += self.g*ad
        self.utilities()
        aas = self.aas()
        aff_comp = (self.w1*ad_sum) + self.w2*aas
        return aff_comp, ad, aas