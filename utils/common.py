def flatten_list(lst):
    """
    e.g.
        lst = [1, [2, 3], [4, [[5, 6], 7]], 8, [9, [10]]]
        flatten_list(lst) => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    return [ele for sublst in lst for ele in flatten_list(sublst)] if isinstance(lst, list) else [lst]
