from typing import List, Tuple

def limited_depth_search_best_model(steps_ahead: int, 
                                    sorted_arrows: List[Tuple], 
                                    is_dag: callable, 
                                    get_score: callable) -> Tuple[float, List[Tuple]]:
    """
    Perform limited-depth search to find the best model and score based on the given arrows.

    Args:
        steps_ahead (int): How many steps ahead to look when making decisions at current step.
        sorted_arrows (List[Tuple]): List of arrows sorted by priority.
        is_dag (callable): Checks if a set of arrows forms a DAG.
        get_score (callable): Returns a numeric score for a set of arrows.
        
    Returns:
        Tuple[float, List[Tuple]]: Best score and corresponding set of arrows.
    """

    def dfs(current_arrows: List[Tuple], index: int, depth_left: int) -> Tuple[float, List[Tuple]]:
        if index >= len(sorted_arrows) or depth_left == 0:
            return get_score(current_arrows), current_arrows

        # Option 1: Skip current arrow
        best_score, best_model = dfs(current_arrows, index + 1, depth_left - 1)

        # Option 2: Include current arrow (if still a DAG)
        next_arrow = sorted_arrows[index]
        with_next = current_arrows + [next_arrow]
        if is_dag(with_next):
            cand_score, cand_model = dfs(with_next, index + 1, depth_left - 1)
            if cand_score > best_score:
                return cand_score, cand_model

        return best_score, best_model

    return dfs([], 0, steps_ahead)
