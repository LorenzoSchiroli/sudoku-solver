import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'sudoku_bridge.dart';

void main() => runApp(const MaterialApp(home: SudokuScreen()));

class SudokuScreen extends StatefulWidget {
  const SudokuScreen({super.key});
  @override
  State<SudokuScreen> createState() => _SudokuScreenState();
}

class _SudokuScreenState extends State<SudokuScreen> {
  final bridge = SudokuBridge();
  final ImagePicker _picker = ImagePicker();
  List<Map<String, int>> sudokuBoard = [];
  bool isProcessing = false;
  bool isInitialized = false;

  @override
  void initState() {
    super.initState();
    // Start initializing the C++ processor immediately
    _setupBridge();
  }

  Future<void> _setupBridge() async {
    await bridge.init();
    setState(() => isInitialized = true);
  }

  Future<void> _takePhoto() async {
    if (!isInitialized) return;

    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image == null) return;

    setState(() => isProcessing = true);

    // Process image in C++
    final result = bridge.solveSudoku(image.path);

    setState(() {
      sudokuBoard = result;
      isProcessing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Sudoku Solver')),
      body: Container(
        width: double.infinity,
        color: Colors.grey[200],
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (!isInitialized)
              const Text("Loading AI Model...")
            else if (isProcessing)
              const CircularProgressIndicator()
            else if (sudokuBoard.isEmpty)
              const Text("Tap the camera button to scan a puzzle")
            else
              AspectRatio(aspectRatio: 1, child: _buildSudokuGrid()),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: isInitialized ? _takePhoto : null,
        label: Text(isInitialized ? "Scan Sudoku" : "Initializing..."),
        icon: const Icon(Icons.camera_alt),
        backgroundColor: isInitialized ? null : Colors.grey,
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
