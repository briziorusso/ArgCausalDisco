class TopDiffAggregation:
    def __init__(self) -> None:
        self.name = "TopAggregation"
        pass

    def aggregate_strength(self, attackers, supporters):
        top_attacker = 0
        top_supporter = 0

        for a in attackers:
            if a > top_attacker:
                top_attacker = a
        
        for s in supporters:
            if s > top_supporter:
                top_supporter = s

        return top_supporter - top_attacker

    def __str__(self) -> str:
        return __class__.__name__
