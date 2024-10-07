"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

""" This file contains tree structure functions.
    The functions are used in Preference Dungeon primarily to check whether all contexts
    have possible valid offers.
"""

from typing import List


class TreeNode:
    """
    Node for a tree data structure that stores strings.
    """

    def __init__(self, value: str):
        self.value = value
        self.children: List[TreeNode] = []


def build_combinations_tree(lists: List[List[str]]) -> TreeNode:
    """
    Build a tree representing all combinations of one item each from the given lists.

    Args:
        lists: A list of lists of strings.

    Returns:
        The root node of the constructed tree.
    """
    if not lists:
        return None

    # sort lists from smallest to largest for elimination efficiency
    lists = sorted(lists, key=len)

    root = TreeNode(None)
    for item in lists[0]:
        child = TreeNode(item)
        root.children.append(child)
        build_subtree(child, lists[1:])
    return root


def build_subtree(node: TreeNode, remaining_lists: List[List[str]]) -> None:
    """
    Recursively build subtrees for each possible combination of items from remaining lists.

    Args:
        node: The current node to which the subtrees will be attached.
        remaining_lists: The remaining lists of strings to be combined.
    """
    if not remaining_lists:
        return

    for item in remaining_lists[0]:
        child = TreeNode(item)
        node.children.append(child)
        build_subtree(child, remaining_lists[1:])


def traverse_tree(root: TreeNode, current: List[str] = []) -> None:
    """
    Traverse the tree and print all combinations of one item from each list.

    Args:
        root: The root node of the tree.
        current: A list containing the current path of traversal.
    """
    if not root:
        return

    if root.value is not None:  # Check if the current node has a value
        current.append(root.value)

    if not root.children:
        print(current)
    else:
        for child in root.children:
            traverse_tree(child, current.copy())


def count_leaf_nodes(root: TreeNode) -> int:
    """
    Count the number of leaf nodes in the tree.

    Args:
        root: The root node of the tree.

    Returns:
        The number of leaf nodes in the tree.
    """
    if not root:
        return 0

    if not root.children:
        # If the current node has no children, it's a leaf node.
        return 1

    # Recursively count leaf nodes for each child and sum them up.
    return sum(count_leaf_nodes(child) for child in root.children)


def max_tree_depth(root: TreeNode) -> int:
    """
    Calculate the maximum depth of the tree.

    Args:
        root: The root node of the tree.

    Returns:
        The maximum depth of the tree.
    """
    if not root:
        return 0

    if not root.children:
        return 1

    max_child_depth = 0
    for child in root.children:
        max_child_depth = max(max_child_depth, max_tree_depth(child))

    return max_child_depth + 1


def remove_node(root: TreeNode, target: str) -> TreeNode:
    """
    Remove the node along with all its subtrees if it matches the given string.

    Args:
        root: The root node of the tree.
        target: The string to search for in the tree.

    Returns:
        The root node of the updated tree.
    """
    if not root:
        return None

    if root.value == target:
        # Node matches the target, remove it and return None to indicate removal.
        return None

    # Recursively remove the node from children if present.
    root.children = [remove_node(child, target) for child in root.children]

    # Remove any children that have become None (i.e., removed nodes).
    root.children = [child for child in root.children if child is not None]

    return root


def copy_tree(root: TreeNode) -> TreeNode:
    """
    Create a deep copy of the given tree.

    Args:
        root: The root node of the original tree.

    Returns:
        The root node of the copied tree.
    """
    if not root:
        return None

    new_root = TreeNode(root.value)
    # Recursively copy children for the new tree.
    new_root.children = [copy_tree(child) for child in root.children]
    return new_root
