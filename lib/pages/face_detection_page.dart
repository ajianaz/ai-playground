import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:ai_playground/util/colors.dart';
import 'package:ai_playground/view_model/face_detection_viewmodel.dart';
import 'package:ai_playground/widget/custom_button.dart';
import 'package:ai_playground/widget/face_detector_painter.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

class FaceDetectionPage extends StatefulWidget {
  const FaceDetectionPage({super.key});

  @override
  State<FaceDetectionPage> createState() => _FaceDetectionPageState();
}

class _FaceDetectionPageState extends State<FaceDetectionPage> {
  late FaceDetectionModel viewModel;

  @override
  void initState() {
    viewModel = context.read<FaceDetectionModel>();
    super.initState();
  }

  // This function is no longer needed since we're getting the image size from the viewmodel

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: lightPinkColor,
      appBar: AppBar(
        title: const Text(
          'Face Detection',
          style: TextStyle(color: white),
        ),
        foregroundColor: white,
        backgroundColor: darkPinkColor,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
          child: Column(
            children: [
              Consumer<FaceDetectionModel>(
                builder: (context, viewModel, child) {
                  if (viewModel.isLoading) {
                    return const Center(
                      child: CircularProgressIndicator(color: darkPinkColor),
                    );
                  }

                  if (viewModel.imagePath == null) {
                    return const Center(
                      child: Text(
                        "No image selected. Please select an image to detect people.",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                            fontSize: 20, fontWeight: FontWeight.w400),
                      ),
                    );
                  }

                  return Column(
                    children: [
                      // Display image with bounding boxes
                      Container(
                        margin: const EdgeInsets.only(bottom: 20),
                        child: viewModel.imageSize != null
                            ? Stack(
                                children: [
                                  Image.file(
                                    File(viewModel.imagePath!),
                                    width: double.infinity,
                                    height: 300,
                                    fit: BoxFit.contain,
                                  ),
                                  if (viewModel.faces.isNotEmpty)
                                    Positioned.fill(
                                      child: CustomPaint(
                                        painter: FaceDetectorPainter(
                                          faces: viewModel.faces,
                                          imageSize: viewModel.imageSize!,
                                          widgetSize: Size(
                                            MediaQuery.of(context).size.width -
                                                40,
                                            300,
                                          ),
                                          rotation: viewModel.imageRotation ??
                                              InputImageRotation.rotation0deg,
                                        ),
                                      ),
                                    ),
                                ],
                              )
                            : const CircularProgressIndicator(
                                color: darkPinkColor),
                      ),

                      // Display face count
                      Container(
                        padding: const EdgeInsets.all(15),
                        decoration: BoxDecoration(
                          color: white,
                          borderRadius: BorderRadius.circular(10),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.withOpacity(0.3),
                              spreadRadius: 2,
                              blurRadius: 5,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Text("People detected: ",
                                style: TextStyle(
                                    fontSize: 20, fontWeight: FontWeight.w400)),
                            Text(viewModel.count.toString(),
                                style: const TextStyle(
                                    fontSize: 20, fontWeight: FontWeight.w500)),
                          ],
                        ),
                      ),
                    ],
                  );
                },
              ),
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: InkWell(
                        onTap: () => viewModel.processImage(ImageSource.camera),
                        child: const CustomButton(text: "Capture from Camera")),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: InkWell(
                        onTap: () =>
                            viewModel.processImage(ImageSource.gallery),
                        child: const CustomButton(text: 'Select from Gallery')),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Consumer<FaceDetectionModel>(
                builder: (context, viewModel, child) {
                  return viewModel.imagePath != null
                      ? InkWell(
                          onTap: () => viewModel.reset(),
                          child: const CustomButton(text: 'Reset'),
                        )
                      : const SizedBox.shrink();
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
