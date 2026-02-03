#include "sudoku_solver.hpp"

#include <iostream>
#include <vector>
#include <array>
#include <string_view>
#include <string>
#include <algorithm>
#include <bit> // Requires C++20
#include <chrono>
#include <cstdint>

/**
 * Modern C++ Sudoku Solver
 * * optimizations:
 * 1. Bitmasks: Rows, cols, and boxes use 16-bit integers to track used numbers.
 * 2. MRV Heuristic: Always branches on the cell with the fewest valid options first.
 * 3. Lookup Tables: Pre-computed indices to avoid division/modulo operations in the hot loop.
 * 4. Cache Locality: Uses a flat std::array.
 * * Compile with: g++ -O3 -std=c++20 sudoku_solver.cpp -o solver
 */

SudokuSolver::SudokuSolver() {
    // Precompute box indices to avoid repetitive calculation
    for (int i = 0; i < CELL_COUNT; ++i) {
        int r = i / N;
        int c = i % N;
        box_indices[i] = (r / 3) * 3 + (c / 3);
    }
}

// Main entry point
bool SudokuSolver::solve(const std::vector<std::vector<int>>& input, std::vector<std::vector<int>>& result) {
    if (input.size() != N) return false;
    for (const auto& row : input) if (row.size() != N) return false;

    // Reset state
    grid.fill(0);
    row_mask.fill(0);
    col_mask.fill(0);
    box_mask.fill(0);
    empty_cells.clear();
    empty_cells.reserve(CELL_COUNT);

    // Parse 2D input into flat grid
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int v = input[r][c];
            if (v >= 1 && v <= 9) place(r * N + c, v);
        }
    }

    // Collect remaining empty cells for the backtracking algorithm
    for(int i = 0; i < CELL_COUNT; ++i) {
        if(grid[i] == 0) empty_cells.push_back(i);
    }

    if (solve_recursive(0)) {
        // Convert flat grid to 2D result
        result.assign(N, std::vector<int>(N));
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                result[r][c] = grid[r * N + c];
            return true;
        }
        return false;
    }

void SudokuSolver::print_grid(const Grid& g) {
    for (int r = 0; r < N; ++r) {
        if (r > 0 && r % 3 == 0) std::cout << "------+-------+------\n";
        for (int c = 0; c < N; ++c) {
            if (c > 0 && c % 3 == 0) std::cout << "| ";
            std::cout << (g[r * N + c] == 0 ? '.' : (char)('0' + g[r * N + c])) << " ";
        }
        std::cout << "\n";
    }
}

// Mark a number as used in the bitmasks and grid
void SudokuSolver::place(int idx, int val) {
    int r = idx / N;
    int c = idx % N;
    int b = box_indices[idx];
    uint16_t bit = 1 << (val - 1);
    
    grid[idx] = val;
    row_mask[r] |= bit;
    col_mask[c] |= bit;
    box_mask[b] |= bit;
}

// Unmark a number (backtracking)
void SudokuSolver::remove(int idx, int val) {
    int r = idx / N;
    int c = idx % N;
    int b = box_indices[idx];
    uint16_t bit = 1 << (val - 1);

    grid[idx] = 0;
    row_mask[r] &= ~bit;
    col_mask[c] &= ~bit;
    box_mask[b] &= ~bit;
}

// Get a bitmask of valid moves for a specific cell index
// Returns 9 bits where 1 means "available"
[[nodiscard]] uint16_t SudokuSolver::get_candidates(int idx) const {
    int r = idx / N;
    int c = idx % N;
    int b = box_indices[idx];
    
    // OR the masks together to get used numbers, then NOT to get available
    return ~(row_mask[r] | col_mask[c] | box_mask[b]) & 0x1FF;
}

// Recursive backtracking with MRV (Minimum Remaining Values) heuristic
// k is the number of filled cells in the *empty_cells* vector we've processed so far
// However, for efficiency with swapping, we pass the current index in the empty_cells vector
bool SudokuSolver::solve_recursive(size_t k) {
    if (k == empty_cells.size()) {
        return true; // All cells filled
    }

    // MRV Heuristic:
    // Instead of just taking empty_cells[k], scan empty_cells[k...end]
    // to find the one with the FEWEST candidates. Swap it to position k.
    size_t best_idx = k;
    int min_candidates = 10;
    uint16_t best_mask = 0;

    for (size_t i = k; i < empty_cells.size(); ++i) {
        uint16_t mask = get_candidates(empty_cells[i]);
        int count = std::popcount(mask);
        
        if (count == 0) return false; // Dead end
        
        if (count < min_candidates) {
            min_candidates = count;
            best_mask = mask;
            best_idx = i;
            if (count == 1) break; // Can't get better than 1
        }
    }

    // Swap the best cell to the current processing position
    std::swap(empty_cells[k], empty_cells[best_idx]);
    
    int current_cell_idx = empty_cells[k];

    // Iterate through the bits set in best_mask
    while (best_mask) {
        // Get index of the lowest set bit (trailing zeros)
        int bit_idx = std::countr_zero(best_mask); 
        int val = bit_idx + 1;

        place(current_cell_idx, val);
        
        if (solve_recursive(k + 1)) {
            return true;
        }

        remove(current_cell_idx, val);
        
        // Clear the lowest set bit to move to the next candidate
        best_mask &= (best_mask - 1);
    }

    return false;
}

// main removed: moved to `test_sudoku_solver.cpp` for library-style usage
