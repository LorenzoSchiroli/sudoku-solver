import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:flutter/foundation.dart';

/// FFI representation of a single Sudoku cell returned from native code.
/// - `number`: the digit in the cell (0 if empty)
/// - `mask`: a flag indicating whether the digit was originally present (1)
final class CellC extends Struct {
  @Int32()
  external int number;
  @Int32()
  external int mask;
}

/// FFI representation of the overall Sudoku result returned by the
/// native processor. `status` is a small integer code and `cells` is
/// an array of 81 `CellC` entries representing the grid.
final class SudokuResultC extends Struct {
  @Int32()
  external int status;
  @Array(81)
  external Array<CellC> cells;
}

/// The native dynamic library handle. On Android this opens the
/// packaged `.so`, on other platforms it uses the current process
/// (useful for desktop embedding during development).
final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libsudoku_bridge.so')
    : DynamicLibrary.process();

// Signatures
typedef CreateProcessorC = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef CreateProcessorDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef ProcessImageBytesC =
    Pointer<SudokuResultC> Function(
      Pointer<Void> processor,
      Pointer<Uint8> data,
      IntPtr length,
    );
typedef ProcessImageBytesDart =
    Pointer<SudokuResultC> Function(
      Pointer<Void> processor,
      Pointer<Uint8> data,
      int length,
    );

typedef FreeResultC = Void Function(Pointer<SudokuResultC> result);
typedef FreeResultDart = void Function(Pointer<SudokuResultC> result);

typedef DebugImageC = Void Function(Pointer<Uint8> data, IntPtr length);
typedef DebugImageDart = void Function(Pointer<Uint8> data, int length);

// Helper Class
class SudokuBridge {
  static final SudokuBridge _instance = SudokuBridge._internal();
  factory SudokuBridge() => _instance;

  Pointer<Void>? _processor;
  late CreateProcessorDart _createFunc;
  late ProcessImageBytesDart _processBytesFunc;
  late FreeResultDart _freeResultFunc;

  SudokuBridge._internal() {
    _createFunc = nativeLib
        .lookupFunction<CreateProcessorC, CreateProcessorDart>(
          'sudoku_create_processor',
        );
    _processBytesFunc = nativeLib
        .lookupFunction<ProcessImageBytesC, ProcessImageBytesDart>(
          'sudoku_process_image_bytes',
        );
    _freeResultFunc = nativeLib.lookupFunction<FreeResultC, FreeResultDart>(
      'sudoku_free_result',
    );
  }

  /// Extracts the model asset into the application's documents
  /// directory (if not already present) and initializes the native
  /// processor by passing the extracted model path to the C++ side.
  Future<void> init() async {
    if (_processor != null) return;

    final docsDir = await getApplicationDocumentsDirectory();
    final modelPath = p.join(docsDir.path, "resnet18_svhn_int8.onnx");
    final modelFile = File(modelPath);

    // 1. Copy from assets if missing
    if (!await modelFile.exists()) {
      // Adjust this path to match your pubspec.yaml exactly
      final data = await rootBundle.load("assets/resnet18_svhn_int8.onnx");
      await modelFile.writeAsBytes(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      );
    }

    // 2. Pass the extracted path to C++
    final modelPathPtr = modelPath.toNativeUtf8();
    _processor = _createFunc(modelPathPtr);
    calloc.free(modelPathPtr);
  }

  /// Sends the provided image bytes to the native processor and
  /// returns a `Map` containing a `board` (list of 81 cell maps with
  /// `number` and `mask`) and a numeric `status` code. Handles
  /// allocation and deallocation of native memory used for the call.
  Future<Map<String, dynamic>> solveSudokuFromBytes(
    Uint8List imageBytes,
  ) async {
    if (_processor == null || _processor!.address == 0) {
      return {'board': [], 'status': 0};
    }

    // 1. Allocate native memory
    final Pointer<Uint8> buffer = malloc.allocate<Uint8>(imageBytes.length);

    try {
      // 2. Copy the bytes
      final nativeBytes = buffer.asTypedList(imageBytes.length);
      nativeBytes.setAll(0, imageBytes);

      // 3. Call C++
      final Pointer<SudokuResultC> resultPtr = _processBytesFunc(
        _processor!,
        buffer,
        imageBytes.length,
      );

      // 4. Handle the result
      final List<Map<String, int>> board = [];
      int status = -1; // Default error status

      if (resultPtr != nullptr) {
        try {
          final result = resultPtr.ref;
          status = result.status;

          for (int i = 0; i < 81; i++) {
            board.add({
              'number': result.cells[i].number,
              'mask': result.cells[i].mask,
            });
          }
        } finally {
          // 5. CRITICAL: Free the struct allocated by C++
          // (assuming your C++ used 'new' to create the result)
          _freeResultFunc(resultPtr);
        }
      }

      return {'board': board, 'status': status};
    } catch (e) {
      debugPrint("Sudoku Solver Error: $e");
      return {'board': [], 'status': -1};
    } finally {
      // 6. ALWAYS free the input image buffer
      malloc.free(buffer);
    }
  }
}
