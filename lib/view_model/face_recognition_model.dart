import 'dart:developer';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:ai_playground/util/sqlite_helper.dart';
import 'package:flutter/foundation.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../model/face_model.dart';

class FaceRecognitionModel extends ChangeNotifier {
  final String modelName = 'assets/mobilefacenet.tflite';
  Interpreter? _interpreter;
  double threshold = 0.5;
  List<FaceModel> knownFaces = [];
  SqliteHelper sqliteHelper = SqliteHelper();
  String? loadingMessage;
  String? personDetected;
  String? errorMessage;
  String? croppedFaceImagePath;

  // Face detector instance
  late final FaceDetector _faceDetector;

  FaceRecognitionModel() {
    // Initialize face detector with tracking enabled
    final options = FaceDetectorOptions(
      enableClassification: true,
      enableLandmarks: true,
      enableContours: false,
      enableTracking: true,
      minFaceSize: 0.1,
      performanceMode: FaceDetectorMode.accurate,
    );
    _faceDetector = FaceDetector(options: options);

    loadModel();
  }

  void setLoadingMessage(String? value) {
    loadingMessage = value;
    notifyListeners();
  }

  void setErrorMessage(String? value) {
    errorMessage = value;
    notifyListeners();
  }

  void clearMessages() {
    loadingMessage = null;
    errorMessage = null;
    // Don't clear personDetected here as we want to keep the recognition result
    notifyListeners();
  }

  void clearAllMessages() {
    loadingMessage = null;
    errorMessage = null;
    personDetected = null;
    croppedFaceImagePath = null;
    notifyListeners();
  }

  // Load the TensorFlow Lite model
  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(modelName);
    knownFaces = await sqliteHelper.readAll();
    List<int>? inputShape = _interpreter!.getInputTensor(0).shape;
    TensorType inputType = _interpreter!.getInputTensor(0).type;
    List<int>? outputShape = _interpreter!.getOutputTensor(0).shape;
    TensorType outputType = _interpreter!.getOutputTensor(0).type;
    notifyListeners();
    log("Model loaded successfully");
    log("Model Input Shape: $inputShape");
    log("Model Input Type: $inputType");
    log("Model Output Shape: $outputShape");
    log("Model Output Type: $outputType");
  }

  Future<void> saveImage(String name, ImageSource source) async {
    clearAllMessages();
    setLoadingMessage("Saving image data to the database\nPlease wait...");
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? imageFile = await picker.pickImage(source: source);
      if (imageFile != null) {
        final Float32List processedImage =
            await preprocessImageWithFaceCrop(imageFile);
        final Float32List? outputVector = await runModel(processedImage);
        if (outputVector != null) {
          FaceModel faceModel = FaceModel(name: name, faceData: outputVector);
          await sqliteHelper.add(faceModel);
          knownFaces.add(faceModel);
          setLoadingMessage("Image saved successfully for $name");
        }
      } else {
        setErrorMessage("No image selected");
      }
    } catch (e) {
      setErrorMessage("Error: ${e.toString()}");
    }
  }

  Future<void> deleteAllData() async {
    clearAllMessages();
    setLoadingMessage("Deleting data...Please wait");
    try {
      await sqliteHelper.clear();
      setLoadingMessage("All data deleted successfully");
    } catch (e) {
      setErrorMessage("Error deleting data: ${e.toString()}");
    }
  }

  // Pick image from camera or gallery and process it
  Future<void> processImage(ImageSource source) async {
    clearAllMessages();
    setLoadingMessage("Processing image...\nPlease wait...");
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? imageFile = await picker.pickImage(source: source);
      if (imageFile != null) {
        final Float32List processedImage =
            await preprocessImageWithFaceCrop(imageFile);
        final Float32List? outputVector = await runModel(processedImage);
        if (outputVector != null) {
          personDetected = recognizeFace(outputVector, knownFaces, threshold);
          notifyListeners();
        }
      } else {
        setErrorMessage("No image selected");
      }
    } catch (e) {
      setErrorMessage("Error: ${e.toString()}");
    } finally {
      setLoadingMessage(null);
    }
  }

  Future<Float32List> preprocessImageWithFaceCrop(XFile imageFile) async {
    final File file = File(imageFile.path);
    final Uint8List imageBytes = await file.readAsBytes();

    img.Image? originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) throw Exception("Failed to decode image.");

    // Convert image to InputImage format for ML Kit
    final inputImage = _convertToInputImage(imageFile);

    // Perform face detection
    final List<Face> faces = await _faceDetector.processImage(inputImage);

    // Validate face detection results
    if (faces.isEmpty) {
      throw Exception("No face detected in image");
    }

    if (faces.length > 1) {
      throw Exception("Multiple faces detected. Only one face is allowed");
    }

    // Get the first detected face
    final Face face = faces.first;
    final ui.Rect boundingBox = face.boundingBox;

    // Validate and adjust bounding box coordinates
    final int imageWidth = originalImage.width;
    final int imageHeight = originalImage.height;

    // ML Kit provides absolute coordinates already, not normalized
    // The boundingBox.left, top, width, height are already in the image's coordinate system

    // Add padding around the bounding box to ensure we capture the entire face
    // This helps with cases where the bounding box is too tight
    const double paddingPercent = 0.15; // 15% padding

    final int x = (boundingBox.left - boundingBox.width * paddingPercent)
        .round()
        .clamp(0, imageWidth - 1);
    final int y = (boundingBox.top - boundingBox.height * paddingPercent)
        .round()
        .clamp(0, imageHeight - 1);
    final int width = (boundingBox.width * (1 + 2 * paddingPercent))
        .round()
        .clamp(1, imageWidth - x);
    final int height = (boundingBox.height * (1 + 2 * paddingPercent))
        .round()
        .clamp(1, imageHeight - y);

    // Validate crop coordinates before cropping
    if (x < 0 || y < 0 || x + width > imageWidth || y + height > imageHeight) {
      throw Exception("Face bounding box is outside image boundaries");
    }

    // Crop the face region
    img.Image? faceImage;
    try {
      faceImage = img.copyCrop(
        originalImage,
        x: x,
        y: y,
        width: width,
        height: height,
      );
    } catch (e) {
      throw Exception("Failed to crop face region: ${e.toString()}");
    }

    // img.copyCrop won't return null, but we'll keep this check for safety
    if (faceImage == null) {
      throw Exception("Failed to crop face from image");
    }

    // Save the cropped face image for display in UI
    try {
      final croppedFaceFile = await _saveCroppedFaceImage(faceImage);
      croppedFaceImagePath = croppedFaceFile.path;
      notifyListeners();
    } catch (e) {
      log("Error saving cropped face image: $e");
      // Continue even if saving fails
    }

    // Resize the face image to 112x112
    img.Image resizedImage = img.copyResize(faceImage, width: 112, height: 112);

    // Convert to Float32List for model input
    Float32List processedImage = Float32List(112 * 112 * 3);
    int pixelIndex = 0;

    for (int y = 0; y < 112; y++) {
      for (int x = 0; x < 112; x++) {
        img.Pixel pixel = resizedImage.getPixel(x, y);
        processedImage[pixelIndex++] = pixel.r.toInt() / 255.0;
        processedImage[pixelIndex++] = pixel.g.toInt() / 255.0;
        processedImage[pixelIndex++] = pixel.b.toInt() / 255.0;
      }
    }

    return processedImage;
  }

  // Helper method to save cropped face image
  Future<File> _saveCroppedFaceImage(img.Image faceImage) async {
    final directory = await getApplicationDocumentsDirectory();
    final String fileName =
        'cropped_face_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final File file = File('${directory.path}/$fileName');

    // Convert image to bytes and save
    final Uint8List imageBytes = img.encodeJpg(faceImage, quality: 90);
    await file.writeAsBytes(imageBytes);
    log("Saved cropped face to: ${file.path}");

    return file;
  }

  // Helper method to convert XFile to InputImage
  InputImage _convertToInputImage(XFile imageFile) {
    final path = imageFile.path;
    final file = File(path);

    // Create InputImage from file path
    return InputImage.fromFilePath(path);
  }

  Float32List normalizeEmbedding(Float32List embedding) {
    double norm = math.sqrt(embedding.fold(0.0, (sum, val) => sum + val * val));
    return Float32List.fromList(embedding.map((val) => val / norm).toList());
  }

  Future<Float32List?> runModel(Float32List inputImage) async {
    try {
      if (_interpreter == null) {
        return null;
      }
      var input = inputImage.reshape([1, 112, 112, 3]);
      var output = List.filled(1 * 192, 0.0).reshape([1, 192]);
      _interpreter?.run(input, output);
      Float32List outputList =
          Float32List.fromList(output.expand<double>((e) => e).toList());
      Float32List normalizedOutput = normalizeEmbedding(outputList);
      
      // Log embedding results
      log("Face embedding generated successfully");
      log("Embedding length: ${normalizedOutput.length}");
      log("First 10 values: ${normalizedOutput.take(10).join(", ")}");
      log("Last 10 values: ${normalizedOutput.skip(normalizedOutput.length - 10).join(", ")}");
      log("Embedding norm: ${math.sqrt(normalizedOutput.fold(0.0, (sum, val) => sum + val * val))}");
      
      return normalizedOutput;
    } catch (e, stackTrace) {
      log("Error running model: $e");
      log("Stack trace: $stackTrace");
    }
    return null;
  }

  double euclideanDistance(List<double> vector1, List<double> vector2) {
    double sum = 0.0;
    for (int i = 0; i < vector1.length; i++) {
      sum += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
    }
    return math.sqrt(sum);
  }

  String recognizeFace(List<double> newFaceVector, List<FaceModel> knownFaces,
      double threshold) {
    String recognizedLabel = "Unknown";
    double minDistance = double.infinity;

    for (FaceModel faceModel in knownFaces) {
      double distance = euclideanDistance(newFaceVector, faceModel.faceData!);
      if (distance < minDistance && distance < threshold) {
        minDistance = distance;
        recognizedLabel = faceModel.name!;
      }
    }
    return recognizedLabel;
  }

  // Dispose method to clean up resources
  @override
  void dispose() {
    _faceDetector.close();
    _interpreter?.close();
    super.dispose();
  }
}
