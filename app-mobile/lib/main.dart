import 'package:flutter/material.dart';
import 'sudoku_bridge.dart';
import 'camera.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(home: SudokuScreen()));
}

class SudokuScreen extends StatefulWidget {
  const SudokuScreen({super.key});
  @override
  State<SudokuScreen> createState() => _SudokuScreenState();
}

class _SudokuScreenState extends State<SudokuScreen> {
  final bridge = SudokuBridge();

  // App State
  List<Map<String, int>> sudokuBoard = [];
  String statusMessage = "";

  // Flags
  bool isBridgeReady = false; // C++ model loaded?

  @override
  void initState() {
    super.initState();
    _setupBridge();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _setupBridge() async {
    await bridge.init();
    if (mounted) setState(() => isBridgeReady = true);
  }

  // Open camera page and await result
  void _startScanning() async {
    final result = await Navigator.of(context).push(
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) =>
            CameraPage(bridge: bridge),
        transitionDuration: Duration.zero,
        reverseTransitionDuration: Duration.zero,
      ),
    );

    if (result != null && mounted) {
      setState(() {
        sudokuBoard = List<Map<String, int>>.from(
          (result['board'] as List).cast<Map<String, int>>(),
        );
        statusMessage = _statusText(result['status'] as int);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // Main/Home/Solution screen

    // 2. This is now BOTH your Home and Solution screen
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sudoku Solver'),
        // Optional: clear the current board to start fresh
        actions: sudokuBoard.isNotEmpty
            ? [
                IconButton(
                  icon: const Icon(Icons.delete),
                  onPressed: () => setState(() => sudokuBoard = []),
                ),
              ]
            : [],
      ),
      body: Container(
        width: double.infinity,
        color: Colors.grey[200],
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (sudokuBoard.isEmpty) ...[
              // HOME STATE: No puzzle scanned yet
              const Icon(Icons.grid_on, size: 100, color: Colors.grey),
              const SizedBox(height: 20),
              Text(
                isBridgeReady ? "Ready to solve!" : "Loading AI Model...",
                style: const TextStyle(fontSize: 18, color: Colors.black54),
              ),
            ] else ...[
              // SOLUTION STATE: Show the grid and status
              Text(
                statusMessage,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              AspectRatio(aspectRatio: 1, child: _buildSudokuGrid()),
            ],
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: isBridgeReady ? _startScanning : null,
        label: Text(sudokuBoard.isEmpty ? "Scan Sudoku" : "Scan Another"),
        icon: const Icon(Icons.camera_alt),
        backgroundColor: isBridgeReady ? null : Colors.grey,
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  Widget _buildSudokuGrid() {
    return Container(
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.black, width: 2),
      ),
      child: GridView.builder(
        physics: const NeverScrollableScrollPhysics(),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 9,
        ),
        itemCount: 81,
        itemBuilder: (context, index) {
          int row = index ~/ 9;
          int col = index % 9;
          final cellData = sudokuBoard[index];
          final number = cellData['number'];
          final mask = cellData['mask'];
          final color = mask == 1 ? Colors.black : Colors.blue;

          return Container(
            decoration: BoxDecoration(
              color: Colors.white,
              border: Border(
                bottom: BorderSide(
                  width: (row % 3 == 2 && row < 8) ? 3.0 : 0.5,
                ),
                right: BorderSide(width: (col % 3 == 2 && col < 8) ? 3.0 : 0.5),
              ),
            ),
            child: Center(
              child: Text(
                number != 0 ? "$number" : "",
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

String _statusText(int status) {
  switch (status) {
    case 0:
      return "Grid not found";
    case 1:
      return "Grid found but not solved";
    case 2:
      return "Solved";
    default:
      return "Unknown status ($status)";
  }
}
