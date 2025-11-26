import 'dart:developer';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:ai_playground/util/sqlite_helper.dart';
import 'package:flutter/foundation.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../model/face_model.dart';

class FaceRecognitionModel extends ChangeNotifier {
  final String modelName = 'assets/mobilefacenet.tflite';
  Interpreter? _interpreter;
  double threshold = 0.9;
  List<FaceModel> knownFaces = [];
  SqliteHelper sqliteHelper = SqliteHelper();
  String? loadingMessage;
  String? personDetected;
  String? errorMessage;

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
        final Float32List processedImage = await preprocessImage(imageFile);
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
        final Float32List processedImage = await preprocessImage(imageFile);
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

  Future<Float32List> preprocessImage(XFile imageFile) async {
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

    // Convert ML Kit coordinates to image coordinates
    // ML Kit uses normalized coordinates (0.0 to 1.0)
    final int x =
        (boundingBox.left * imageWidth).round().clamp(0, imageWidth - 1);
    final int y =
        (boundingBox.top * imageHeight).round().clamp(0, imageHeight - 1);
    final int width =
        (boundingBox.width * imageWidth).round().clamp(1, imageWidth - x);
    final int height =
        (boundingBox.height * imageHeight).round().clamp(1, imageHeight - y);

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

    if (faceImage == null) {
      throw Exception("Failed to crop face from image");
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
      _interpreter?.close();
      _interpreter = await Interpreter.fromAsset(modelName);

      if (_interpreter == null) {
        return null;
      }
      var input = inputImage.reshape([1, 112, 112, 3]);
      var output = List.filled(1 * 192, 0.0).reshape([1, 192]);
      _interpreter?.run(input, output);
      Float32List outputList =
          Float32List.fromList(output.expand<double>((e) => e).toList());
      Float32List normalizedOutput = normalizeEmbedding(outputList);
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
