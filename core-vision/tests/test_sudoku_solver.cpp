#include "sudoku_solver.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    SudokuSolver solver;

    // "AI Escargot" - widely considered one of the hardest Sudokus for computers
    // 0 represents an empty cell.
    std::vector<std::vector<int>> puzzle = {
        {1,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,5,0,0},
        {0,0,0,0,0,0,0,0,0},
        {0,0,0,0,8,0,0,0,2},
        {6,0,0,0,0,4,0,0,0},
        {0,0,0,0,0,0,0,1,0},
        {0,4,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0},
    };

    std::cout << "Solving puzzle:\n";

    // Print initial state by converting to flat grid
    SudokuSolver::Grid initial_flat;
    for (int r = 0; r < SudokuSolver::N; ++r)
        for (int c = 0; c < SudokuSolver::N; ++c)
            initial_flat[r * SudokuSolver::N + c] = puzzle[r][c];
    SudokuSolver::print_grid(initial_flat);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int>> result2d;
    bool found = solver.solve(puzzle, result2d);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "\nStatus: " << (found ? "Solved" : "Unsolvable") << "\n";
    std::cout << "Time: " << elapsed.count() << " microseconds\n\n";

    if (found) {
        SudokuSolver::Grid result_flat;
        for (int r = 0; r < SudokuSolver::N; ++r)
            for (int c = 0; c < SudokuSolver::N; ++c)
                result_flat[r * SudokuSolver::N + c] = result2d[r][c];
        SudokuSolver::print_grid(result_flat);
    }

    return found ? 0 : 1;
}
