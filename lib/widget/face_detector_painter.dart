import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceDetectorPainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final Size widgetSize;
  final InputImageRotation rotation;

  FaceDetectorPainter({
    required this.faces,
    required this.imageSize,
    required this.widgetSize,
    required this.rotation,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.red;

    // Calculate the scale factor to fit the image within the widget
    final double scaleX = widgetSize.width / imageSize.width;
    final double scaleY = widgetSize.height / imageSize.height;
    final double scale = scaleX < scaleY ? scaleX : scaleY;

    // Calculate the actual image size after scaling
    final Size scaledImageSize = Size(
      imageSize.width * scale,
      imageSize.height * scale,
    );

    // Calculate the offset to center the image
    final Offset offset = Offset(
      (widgetSize.width - scaledImageSize.width) / 2,
      (widgetSize.height - scaledImageSize.height) / 2,
    );

    for (final Face face in faces) {
      final Rect faceRect = _scaleRect(
        rect: face.boundingBox,
        imageSize: imageSize,
        widgetSize: widgetSize,
        rotation: rotation,
        scale: scale,
        offset: offset,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(faceRect, const Radius.circular(10)),
        paint,
      );
    }
  }

  Rect _scaleRect({
    required Rect rect,
    required Size imageSize,
    required Size widgetSize,
    required InputImageRotation rotation,
    required double scale,
    required Offset offset,
  }) {
    // Scale the rect based on the image to widget size ratio
    Rect scaledRect = Rect.fromLTRB(
      rect.left * scale,
      rect.top * scale,
      rect.right * scale,
      rect.bottom * scale,
    );

    // Apply the offset to center the bounding box
    scaledRect = Rect.fromLTRB(
      scaledRect.left + offset.dx,
      scaledRect.top + offset.dy,
      scaledRect.right + offset.dx,
      scaledRect.bottom + offset.dy,
    );

    // Handle rotation if needed (for simplicity, we're assuming no rotation for now)
    switch (rotation) {
      case InputImageRotation.rotation90deg:
        // Handle 90-degree rotation
        break;
      case InputImageRotation.rotation270deg:
        // Handle 270-degree rotation
        break;
      case InputImageRotation.rotation180deg:
        // Handle 180-degree rotation
        break;
      case InputImageRotation.rotation0deg:
      default:
        // No rotation needed
        break;
    }

    return scaledRect;
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.faces != faces;
  }
}
