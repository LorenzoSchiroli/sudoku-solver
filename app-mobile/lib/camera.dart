import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:typed_data';
import 'sudoku_bridge.dart';

class CameraPage extends StatefulWidget {
  final SudokuBridge bridge;
  final List<CameraDescription>? cameras;

  const CameraPage({super.key, required this.bridge, this.cameras});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _controller;
  bool isProcessing = false;
  bool isFlashOn = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _startCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _controller?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _startCamera();
    }
  }

  Future<void> _startCamera() async {
    List<CameraDescription> available = widget.cameras ?? [];
    if (available.isEmpty) {
      try {
        available = await availableCameras();
      } on CameraException catch (e) {
        print('Error initializing camera: $e');
        return;
      }
    }

    if (available.isEmpty) return;

    final controller = CameraController(
      available[0],
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    _controller = controller;

    try {
      await controller.initialize();
      await controller.setFlashMode(FlashMode.off);
      if (mounted) setState(() => isFlashOn = false);
    } catch (e) {
      print("Camera init error: $e");
    }
  }

  Future<void> _stopCamera() async {
    await _controller?.dispose();
    _controller = null;
    if (mounted) Navigator.of(context).pop();
  }

  Future<void> _toggleFlash() async {
    if (_controller == null) return;
    try {
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

      final result = await widget.bridge.solveSudokuFromBytes(imageBytes);

      if (mounted) {
        Navigator.of(context).pop(result);
      }
    } catch (e) {
      print("Error capturing: $e");
      await _controller?.dispose();
      _controller = null;
      if (mounted) Navigator.of(context).pop();
    } finally {
      if (mounted) setState(() => isProcessing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final bool isControllerReady =
        _controller != null && _controller!.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          if (isControllerReady)
            Center(child: CameraPreview(_controller!))
          else
            const Center(child: CircularProgressIndicator(color: Colors.white)),

          Center(
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                border: Border.all(
                  color: Color.fromRGBO(255, 255, 255, 0.5),
                  width: 2,
                ),
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),

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

          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 30),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    const SizedBox(width: 50),
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
}
