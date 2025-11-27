import 'package:camera/camera.dart';

class ImageUtils {
  /// Convert YUV420_888 (Android camera format) to NV21 format
  static Uint8List convertYUV420ToNV21(
    CameraImage image,
    Uint8List yPlane,
    Uint8List uPlane,
    Uint8List vPlane,
  ) {
    final width = image.width;
    final height = image.height;
    
    final yRowStride = image.planes[0].bytesPerRow;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;
    
    final nv21 = Uint8List(width * height * 3 ~/ 2);
    
    // Copy Y plane
    int uvIndex = 0;
    for (int y = 0; y < height; ++y) {
      final yRowStart = y * width;
      final yRowOffset = y * yRowStride;
      
      // Copy Y row
      for (int x = 0; x < width; ++x) {
        nv21[yRowStart + x] = yPlane[yRowOffset + x];
      }
      
      // Copy UV row (only on even rows)
      if (y % 2 == 0) {
        final uvRowStart = width * height + y ~/ 2 * width;
        final uvRowOffset = (y ~/ 2) * uvRowStride;
        
        for (int x = 0; x < width; x += 2) {
          uvIndex = uvRowStart + x;
          final uvPixelIndex = uvRowOffset + (x ~/ 2) * uvPixelStride;
          
          // V value comes first in NV21 format
          nv21[uvIndex] = vPlane[uvPixelIndex];
          nv21[uvIndex + 1] = uPlane[uvPixelIndex];
        }
      }
    }
    
    return nv21;
  }
}
