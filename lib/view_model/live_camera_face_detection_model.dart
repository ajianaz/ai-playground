import 'dart:developer';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class LiveCameraFaceDetectionModel extends ChangeNotifier {
  CameraController? cameraController;
  List<CameraDescription> cameras = [];
  bool isCameraInitialized = false;
  List<Face> faces = [];
  bool isProcessing = false;
  String? detectedPerson;
  bool isDetecting = false;
  String? errorMessage;
  int currentCameraIndex = 0; // 0 for back camera, 1 for front camera

  // Face detector instance
  late final FaceDetector _faceDetector;

  LiveCameraFaceDetectionModel() {
    // Initialize face detector
    final options = FaceDetectorOptions(
      enableClassification: true,
      enableLandmarks: true,
      enableContours: false,
      enableTracking: true,
      minFaceSize: 0.1,
      performanceMode: FaceDetectorMode.fast,
    );
    _faceDetector = FaceDetector(options: options);
  }

  Future<void> initializeCamera() async {
    try {
      cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        _initializeCameraController(currentCameraIndex);
      }
    } catch (e) {
      log("Error initializing camera: $e");
      errorMessage = "Error initializing camera: $e";
      notifyListeners();
    }
  }

  Future<void> _initializeCameraController(int cameraIndex) async {
    try {
      // Dispose previous controller if exists
      await cameraController?.dispose();
      
      cameraController = CameraController(
        cameras[cameraIndex],
        ResolutionPreset.high,
        enableAudio: false,
      );
      
      await cameraController!.initialize();
      
      isCameraInitialized = true;
      notifyListeners();
      
      // Start face detection stream
      cameraController!.startImageStream((CameraImage cameraImage) {
        if (!isDetecting) {
          isDetecting = true;
          detectFaces(cameraImage);
        }
      });
    } catch (e) {
      log("Error initializing camera controller: $e");
      errorMessage = "Error initializing camera controller: $e";
      notifyListeners();
    }
  }

  Future<void> switchCamera() async {
    if (cameras.length < 2) return; // No need to switch if only one camera
    
    // Switch camera index
    currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
    notifyListeners();
    
    // Reinitialize camera with new index
    await _initializeCameraController(currentCameraIndex);
  }

  Future<void> detectFaces(CameraImage cameraImage) async {
    try {
      final inputImage = _inputImageFromCameraImage(cameraImage);
      if (inputImage != null) {
        log("Processing image for face detection...");
        faces = await _faceDetector.processImage(inputImage);
        log("Face detection result: ${faces.length} faces detected");
        
        if (faces.isNotEmpty && faces.length == 1 && !isProcessing) {
          // Only process single face and avoid multiple simultaneous processes
          isProcessing = true;
          
          // For now, let's just show that we detected a face
          detectedPerson = "Face detected";
          notifyListeners();
          
          // Reset processing flag after a delay
          Future.delayed(const Duration(seconds: 1), () {
            isProcessing = false;
          });
        } else if (faces.length > 1) {
          detectedPerson = "Multiple faces detected";
          notifyListeners();
        } else {
          detectedPerson = null;
          notifyListeners();
        }
      } else {
        log("Failed to create input image from camera image");
      }
    } catch (e) {
      log("Error detecting faces: $e");
      errorMessage = "Error detecting faces: $e";
      notifyListeners();
    } finally {
      isDetecting = false;
      notifyListeners();
    }
  }

  InputImage? _inputImageFromCameraImage(CameraImage cameraImage) {
    if (cameraController == null) return null;

    try {
      final camera = cameraController!.description;
      
      // Get device rotation (assuming portrait mode for mobile devices)
      InputImageRotation rotation;
      if (camera.lensDirection == CameraLensDirection.front) {
        // Front camera
        rotation = InputImageRotation.rotation270deg;
      } else {
        // Back camera
        rotation = InputImageRotation.rotation90deg;
      }

      // Convert YUV420 format to NV21 which ML Kit expects
      // This is the standard approach for Android camera images
      final int width = cameraImage.width;
      final int height = cameraImage.height;
      final int uvRowStride = cameraImage.planes[1].bytesPerRow;
      final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;
      
      final yPlane = cameraImage.planes[0].bytes;
      final uPlane = cameraImage.planes[1].bytes;
      final vPlane = cameraImage.planes[2].bytes;
      
      final nv21 = Uint8List(width * height + (width ~/ 2) * (height ~/ 2) * 2);
      
      // Copy Y plane
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          nv21[y * width + x] = yPlane[y * width + x];
        }
      }
      
      // Copy UV planes (interleaved VU for NV21)
      int uvIndex = 0;
      for (int y = 0; y < height ~/ 2; y++) {
        for (int x = 0; x < width ~/ 2; x++) {
          final uvPixelIndex = y * uvRowStride + x * uvPixelStride;
          nv21[width * height + uvIndex] = vPlane[uvPixelIndex];
          nv21[width * height + uvIndex + 1] = uPlane[uvPixelIndex];
          uvIndex += 2;
        }
      }
      
      return InputImage.fromBytes(
        bytes: nv21,
        metadata: InputImageMetadata(
          size: Size(width.toDouble(), height.toDouble()),
          rotation: rotation,
          format: InputImageFormat.nv21,
          bytesPerRow: width,
        ),
      );
    } catch (e) {
      log("Error converting camera image: $e");
      return null;
    }
  }

  @override
  void dispose() {
    cameraController?.dispose();
    _faceDetector.close();
    super.dispose();
  }
}
