import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image_picker/image_picker.dart';

class FaceDetectionModel extends ChangeNotifier {
  int? count;
  bool isLoading = false;
  String? imagePath;
  List<Face> faces = [];
  Size? imageSize;
  InputImageRotation? imageRotation;
  final FaceDetector faceDetector =
      FaceDetector(options: FaceDetectorOptions(enableTracking: true));

  Future<void> processImage(ImageSource source) async {
    final ImagePicker picker = ImagePicker();
    final XFile? photo = await picker.pickImage(source: source);
    setLoading(true);
    await detectFaces(photo);
    setLoading(false);
  }

  Future<void> detectFaces(XFile? imageFile) async {
    if (imageFile == null) return;

    // Store the image path
    imagePath = imageFile.path;

    // Get image size
    final File image = File(imageFile.path);
    final Uint8List bytes = await image.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    imageSize =
        Size(frame.image.width.toDouble(), frame.image.height.toDouble());

    // Determine image rotation (for simplicity, we'll assume no rotation)
    imageRotation = InputImageRotation.rotation0deg;

    final inputImage = InputImage.fromFilePath(imageFile.path);
    final List<Face> detectedFaces =
        await faceDetector.processImage(inputImage);

    // Store the detected faces
    faces = detectedFaces;
    count = faces.length;
    notifyListeners();
  }

  void setLoading(bool value) {
    isLoading = value;
    notifyListeners();
  }

  void reset() {
    imagePath = null;
    faces = [];
    count = null;
    imageSize = null;
    imageRotation = null;
    notifyListeners();
  }
}
