import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

// 1. Define C Structs
final class CellC extends Struct {
  @Int32()
  external int number;
  @Int32()
  external int mask;
}

final class SudokuResultC extends Struct {
  @Int32()
  external int status;
  @Array(81)
  external Array<CellC> cells;
}

// 2. Load the Library
final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libsudoku_bridge.so')
    : DynamicLibrary.process();

// 3. Signatures
typedef CreateProcessorC = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef CreateProcessorDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef ProcessImageFileC =
    Pointer<SudokuResultC> Function(
      Pointer<Void> processor,
      Pointer<Utf8> path,
    );
typedef ProcessImageFileDart =
    Pointer<SudokuResultC> Function(
      Pointer<Void> processor,
      Pointer<Utf8> path,
    );

typedef FreeResultC = Void Function(Pointer<SudokuResultC> result);
typedef FreeResultDart = void Function(Pointer<SudokuResultC> result);

// 4. Helper Class
class SudokuBridge {
  Pointer<Void>? _processor;
  late CreateProcessorDart _createFunc;
  late ProcessImageFileDart _processFileFunc;
  late FreeResultDart _freeResultFunc;

  SudokuBridge() {
    _createFunc = nativeLib
        .lookupFunction<CreateProcessorC, CreateProcessorDart>(
          'sudoku_create_processor',
        );
    _processFileFunc = nativeLib
        .lookupFunction<ProcessImageFileC, ProcessImageFileDart>(
          'sudoku_process_image_file',
        );
    _freeResultFunc = nativeLib.lookupFunction<FreeResultC, FreeResultDart>(
      'sudoku_free_result',
    );
  }

  /// Extracts the model from assets to local storage and initializes C++
  Future<void> init() async {
    if (_processor != null) return;

    // Path on Android: /data/user/0/com.example.../app_flutter/resnet18_svhn_int8.onnx
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

  List<Map<String, int>> solveSudoku(String imagePath) {
    if (_processor == null) return List.filled(81, {'number': 0, 'mask': 0});

    final pathPtr = imagePath.toNativeUtf8();
    final resultPtr = _processFileFunc(_processor!, pathPtr);

    List<Map<String, int>> board = [];
    if (resultPtr != nullptr) {
      final result = resultPtr.ref;
      for (int i = 0; i < 81; i++) {
        board.add({
          'number': result.cells[i].number,
          'mask': result.cells[i].mask,
        });
      }
      _freeResultFunc(resultPtr);
    }

    calloc.free(pathPtr);
    return board;
  }
}
