def american_to_prob(american_odds):
    """
    Convert American odds to implied probability (decimal)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)

def no_vig_prob(prob):
    """
    Normalize probability removing vig
    """
    return prob / sum(prob) if isinstance(prob, (list, tuple)) else prob

def american_to_decimal(american_odds):
    """
    Convert American odds to decimal odds
    """
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

def decimal_to_american(decimal_odds):
    """
    Convert decimal odds to American
    """
    if decimal_odds >= 2:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)
