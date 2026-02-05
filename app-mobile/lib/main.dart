import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'sudoku_bridge.dart';
import 'dart:typed_data';

// Global variable to store available cameras
List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Error initializing camera: $e');
  }
  runApp(const MaterialApp(home: SudokuScreen()));
}

class SudokuScreen extends StatefulWidget {
  const SudokuScreen({super.key});
  @override
  State<SudokuScreen> createState() => _SudokuScreenState();
}

class _SudokuScreenState extends State<SudokuScreen>
    with WidgetsBindingObserver {
  final bridge = SudokuBridge();
  CameraController? _controller;

  // App State
  List<Map<String, int>> sudokuBoard = [];
  String statusMessage = "";

  // Flags
  bool isBridgeReady = false; // C++ model loaded?
  bool isCameraActive = false; // Are we currently in camera mode?
  bool isProcessing = false; // Are we processing an image?
  bool isFlashOn = false; // Is the flash (torch) enabled?

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _setupBridge();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  // Handle app lifecycle (background/foreground)
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;

    if (state == AppLifecycleState.inactive) {
      _controller?.dispose();
    } else if (state == AppLifecycleState.resumed && isCameraActive) {
      _startCamera(); // Re-initialize if we come back and camera was active
    }
  }

  Future<void> _setupBridge() async {
    await bridge.init();
    if (mounted) setState(() => isBridgeReady = true);
  }

  // --- Camera Logic ---

  Future<void> _startCamera() async {
    if (cameras.isEmpty) return;

    final controller = CameraController(
      cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    _controller = controller;

    try {
      await controller.initialize();
      // IMPORTANT: Turn flash OFF by default to stop random flashing
      await controller.setFlashMode(FlashMode.off);

      if (mounted) {
        setState(() {
          isCameraActive = true;
          isFlashOn = false;
        });
      }
    } catch (e) {
      print("Camera init error: $e");
    }
  }

  Future<void> _stopCamera() async {
    if (mounted) setState(() => isCameraActive = false);
    await _controller?.dispose();
    _controller = null;
  }

  Future<void> _toggleFlash() async {
    if (_controller == null) return;
    try {
      // Toggle between Torch (steady light) and Off
      FlashMode newMode = isFlashOn ? FlashMode.off : FlashMode.torch;
      await _controller!.setFlashMode(newMode);
      setState(() => isFlashOn = !isFlashOn);
    } catch (e) {
      print("Error toggling flash: $e");
    }
  }

  Future<void> _takePhoto() async {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        isProcessing)
      return;

    setState(() => isProcessing = true);

    try {
      final XFile image = await _controller!.takePicture();
      final Uint8List imageBytes = await image.readAsBytes();

      await _controller!.pausePreview();

      final result = await bridge.solveSudokuFromBytes(imageBytes);

      if (mounted) {
        setState(() {
          sudokuBoard = List<Map<String, int>>.from(
            (result['board'] as List).cast<Map<String, int>>(),
          );
          statusMessage = _statusText(result['status'] as int);
          isCameraActive = false;
        });
        await _controller?.dispose();
        _controller = null;
      }
    } catch (e) {
      print("Error capturing: $e");
      _stopCamera(); // Ensure camera closes on error
    } finally {
      if (mounted) setState(() => isProcessing = false);
    }
  }

  // Update this function to jump straight to the camera
  void _startScanning() {
    _startCamera();
  }

  @override
  Widget build(BuildContext context) {
    // 1. If camera is active, show the full-screen camera UI
    if (isCameraActive) {
      return _buildCameraUI(); // I've wrapped your existing camera Stack below
    }

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

  // To keep the build method clean, move the Camera Stack logic here:
  Widget _buildCameraUI() {
    final bool isControllerReady =
        _controller != null && _controller!.value.isInitialized;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          if (isControllerReady && isCameraActive)
            Center(child: CameraPreview(_controller!))
          else
            const Center(child: CircularProgressIndicator(color: Colors.white)),

          // Centered Target Square
          Center(
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                border: Border.all(
                  color: Colors.white.withOpacity(0.5),
                  width: 2,
                ),
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),

          // Close Button
          SafeArea(
            child: Align(
              alignment: Alignment.topLeft,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: IconButton(
                  icon: const Icon(Icons.close, color: Colors.white, size: 30),
                  onPressed: _stopCamera,
                ),
              ),
            ),
          ),

          // Bottom Controls
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 30),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    const SizedBox(width: 50), // Spacer
                    isProcessing
                        ? const CircularProgressIndicator(color: Colors.white)
                        : FloatingActionButton.large(
                            onPressed: _takePhoto,
                            backgroundColor: Colors.white,
                            child: const Icon(
                              Icons.camera_alt,
                              color: Colors.black,
                              size: 40,
                            ),
                          ),
                    SizedBox(
                      width: 50,
                      child: IconButton(
                        icon: Icon(
                          isFlashOn ? Icons.flash_on : Icons.flash_off,
                          color: Colors.white,
                        ),
                        onPressed: _toggleFlash,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
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
