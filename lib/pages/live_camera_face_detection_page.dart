import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:ai_playground/util/colors.dart';
import 'package:ai_playground/view_model/live_camera_face_detection_model.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:camera/camera.dart';

class LiveCameraFaceDetectionPage extends StatefulWidget {
  const LiveCameraFaceDetectionPage({super.key});

  @override
  State<LiveCameraFaceDetectionPage> createState() =>
      _LiveCameraFaceDetectionPageState();
}

class _LiveCameraFaceDetectionPageState
    extends State<LiveCameraFaceDetectionPage> {
  late LiveCameraFaceDetectionModel viewModel;

  @override
  void initState() {
    super.initState();
    viewModel = context.read<LiveCameraFaceDetectionModel>();
    viewModel.initializeCamera();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<LiveCameraFaceDetectionModel>(
      builder: (context, viewModel, child) {
        if (!viewModel.isCameraInitialized) {
          return Scaffold(
            backgroundColor: lightPinkColor,
            appBar: AppBar(
              title: const Text(
                'Live Camera Face Detection',
                style: TextStyle(color: white),
              ),
              foregroundColor: white,
              backgroundColor: darkPinkColor,
            ),
            body: const Center(
              child: CircularProgressIndicator(color: darkPinkColor),
            ),
          );
        }

        return Scaffold(
          backgroundColor: lightPinkColor,
          appBar: AppBar(
            title: const Text(
              'Live Camera Face Detection',
              style: TextStyle(color: white),
            ),
            foregroundColor: white,
            backgroundColor: darkPinkColor,
            actions: [
              IconButton(
                onPressed: () => viewModel.switchCamera(),
                icon: const Icon(Icons.switch_camera, color: white),
              ),
            ],
          ),
          body: Stack(
            children: [
              // Camera preview
              Center(
                child: CameraPreview(viewModel.cameraController!),
              ),
              
              // Face detection overlay
              if (viewModel.faces.isNotEmpty)
                Positioned.fill(
                  child: CustomPaint(
                    painter: FaceDetectorPainter(faces: viewModel.faces),
                  ),
                ),
              
              // Results display
              Positioned(
                bottom: 100,
                left: 20,
                right: 20,
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.7),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Column(
                    children: [
                      if (viewModel.detectedPerson != null)
                        Text(
                          viewModel.detectedPerson!,
                          style: const TextStyle(
                            color: white,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      if (viewModel.faces.isNotEmpty)
                        Text(
                          'Faces detected: ${viewModel.faces.length}',
                          style: const TextStyle(color: white, fontSize: 16),
                        ),
                      if (viewModel.faces.length > 1)
                        const Text(
                          'Multiple faces detected. Please ensure only one face is visible.',
                          style: TextStyle(color: Colors.orange, fontSize: 14),
                          textAlign: TextAlign.center,
                        ),
                    ],
                  ),
                ),
              ),
              
              // Instructions
              if (viewModel.faces.isEmpty)
                Positioned(
                  top: 20,
                  left: 20,
                  right: 20,
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.7),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Text(
                      'Position your face in the camera view',
                      style: TextStyle(color: white, fontSize: 16),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  // Don't dispose the view model here as it's managed by Provider
  // The view model will be properly disposed when the app is closed
  // or when the widget tree that consumes it is unmounted
}

class FaceDetectorPainter extends CustomPainter {
  final List<Face> faces;

  FaceDetectorPainter({required this.faces});

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.green;

    for (final Face face in faces) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(
            face.boundingBox.left,
            face.boundingBox.top,
            face.boundingBox.width,
            face.boundingBox.height,
          ),
          const Radius.circular(10),
        ),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.faces != faces;
  }
}
