#pragma once

#include <array>
#include <vector>

/**
 * Fast backtracking Sudoku solver using bitmasking and MRV heuristic.
 * Public API is `solve`, which accepts a 9x9 integer grid (0 == empty)
 * and writes the solved grid into `result` when successful.
 */
class SudokuSolver {
public:
    static constexpr int N = 9;
    static constexpr int CELL_COUNT = 81;
    using Grid = std::array<int, CELL_COUNT>;

    SudokuSolver();

    /**
     * Solve a Sudoku puzzle.
     * @param input 9x9 grid where 0 indicates an empty cell.
     * @param result Output 9x9 solved grid on success.
     * @return true if solved successfully, false otherwise.
     */
    bool solve(const std::vector<std::vector<int>>& input, std::vector<std::vector<int>>& result);

    /**
     * Utility to print an internal flat grid for debugging.
     */
    static void print_grid(const Grid& g);

private:
    Grid grid;
    std::array<uint16_t, N> row_mask;
    std::array<uint16_t, N> col_mask;
    std::array<uint16_t, N> box_mask;
    std::array<int, CELL_COUNT> box_indices;
    std::vector<int> empty_cells;

    void place(int idx, int val);
    void remove(int idx, int val);
    uint16_t get_candidates(int idx) const;
    bool solve_recursive(size_t k);
    bool is_valid_input(const std::vector<std::vector<int>>& input);
};
