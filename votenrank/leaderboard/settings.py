RANKING_METHODS = ["mean", "plurality", "borda", "dowdall", "copeland", "minimax", "optimality_gap"]

ELECTION_METHODS = [
    "mean",
    "plurality",
    "borda",
    "dowdall",
    "copeland",
    "minimax",
    "threshold",
    "baldwin",
    "condorcet",
    "optimality_gap",
]

PARTIAL_METHODS = ["condorcet", "copeland", "minimax", "mean", "optimality_gap"]

METHODS_SETTINGS = {
    "mean": {"mean_type": ["arithmetic", "geometric"]},
    "copeland": {"slice_type": ["difference"]},
    "minimax": {"score_type": ["winning_votes"]},
    "optimality_gap": {"gamma": [95]}
}

PRETTY_NAMES = {
    "Method: mean, Params: {'mean_type': 'arithmetic'}": "AM",
    "Method: mean, Params: {'mean_type': 'geometric'}": "GM",
    "Method: copeland, Params: {'slice_type': 'difference'}": "Copeland",
    "Method: minimax, Params: {'score_type': 'winning_votes'}": "Minimax",
    "Method: plurality, Params: {}": "Plurality",
    "Method: dowdall, Params: {}": "Dowdall",
    "Method: borda, Params: {}": "Borda",
    "Method: threshold, Params: {}": "Threshold",
    "Method: baldwin, Params: {}": "Baldwin",
    "Method: condorcet, Params: {}": "Condorcet",
    "Method: optimality_gap, Params: {'gamma': 95}": "OG",
}
