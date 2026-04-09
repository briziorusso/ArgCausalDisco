from typing import List, Tuple

def search_best_model_every_step(steps_ahead: int, 
                                 sorted_arrows: List[Tuple], 
                                 is_dag: callable, 
                                 get_score: callable) -> Tuple[float, List[Tuple]]:
    """
    Perform limited-depth search to find the best model and score based on the given arrows.
    Search es performed at each arrow in contrast to a single search at the top arrow.

    Args:
        steps_ahead (int): How many steps ahead to look when making decisions at current step.
        sorted_arrows (List[Tuple]): List of arrows sorted by priority.
        is_dag (callable): Checks if a set of arrows forms a DAG.
        get_score (callable): Returns a numeric score for a set of arrows.
        
    Returns:
        Tuple[float, List[Tuple]]: Best score and corresponding set of arrows.
    """

    def dfs(current_arrows: List[Tuple], next_arrows: List[Tuple], index: int, depth_left: int) -> Tuple[float, List[Tuple]]:
        if index >= len(next_arrows) or depth_left == 0:
            return get_score(current_arrows), current_arrows

        # Option 1: Skip current arrow
        best_score, best_model = dfs(current_arrows, next_arrows, index + 1, depth_left - 1)

        # Option 2: Include current arrow (if still a DAG)
        next_arrow = next_arrows[index]
        with_next = current_arrows + [next_arrow]
        if is_dag(with_next):
            cand_score, cand_model = dfs(with_next, next_arrows, index + 1, depth_left - 1)
            if cand_score > best_score:
                return cand_score, cand_model

        return best_score, best_model

    # loop over each arrow and decide whether to include it or not
    arrows_so_far = []
    for step in range(len(sorted_arrows)):
        arrows_so_far_skip = arrows_so_far.copy()
        arrows_so_far_include = arrows_so_far.copy() + [sorted_arrows[step]]

        if is_dag(arrows_so_far_include):
            best_score_skip, _ = dfs(arrows_so_far_skip, sorted_arrows[step+1:], 0, steps_ahead)
            best_score_include, _ = dfs(arrows_so_far_include, sorted_arrows[step+1:], 0, steps_ahead)
            if best_score_include > best_score_skip:
                arrows_so_far = arrows_so_far_include
            else:
                arrows_so_far = arrows_so_far_skip

    return get_score(arrows_so_far), arrows_so_far
