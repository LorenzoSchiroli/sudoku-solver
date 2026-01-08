#pragma once

#include <array>
#include <vector>

class SudokuSolver {
public:
    static constexpr int N = 9;
    static constexpr int CELL_COUNT = 81;
    using Grid = std::array<int, CELL_COUNT>;

    SudokuSolver(); // Constructor declaration
    bool solve(const std::vector<std::vector<int>>& input, std::vector<std::vector<int>>& result);
    static void print_grid(const Grid& g);

private:
    // Move all private members here
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
};
