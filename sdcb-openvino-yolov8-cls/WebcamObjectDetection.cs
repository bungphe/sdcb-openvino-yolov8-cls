using OpenCvSharp;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;
using Sdcb.OpenVINO.Natives;
using System.Diagnostics;

namespace YoloObjectDetection
{
    public class WebcamObjectDetection
    {
        private readonly string[] _labels;
        private readonly CompiledModel _compiledModel;
        private readonly Model _model;
        private readonly float _confidenceThreshold;
        private readonly float _nmsThreshold;
        private readonly Random _random = new Random();
        private readonly Dictionary<int, Scalar> _colorMap = new Dictionary<int, Scalar>();

        public WebcamObjectDetection(string modelPath, string labelsPath, float confidenceThreshold = 0.25f, float nmsThreshold = 0.45f)
        {
            _confidenceThreshold = confidenceThreshold;
            _nmsThreshold = nmsThreshold;

            // Load labels
            _labels = File.ReadAllLines(labelsPath);
            Console.WriteLine($"Loaded {_labels.Length} class labels");

            // Initialize color map for each class
            for (int i = 0; i < _labels.Length; i++)
            {
                _colorMap[i] = new Scalar(_random.Next(0, 256), _random.Next(0, 256), _random.Next(0, 256));
            }

            // Load and prepare model
            using Model rawModel = OVCore.Shared.ReadModel(modelPath);
            using PrePostProcessor pp = rawModel.CreatePrePostProcessor();

            using (PreProcessInputInfo inputInfo = pp.Inputs.Primary)
            {
                inputInfo.TensorInfo.Layout = Layout.NHWC;
                inputInfo.ModelInfo.Layout = Layout.NCHW;
                inputInfo.TensorInfo.ColorFormat = ov_color_format_e.BGR;
                inputInfo.Steps.ConvertColor(ov_color_format_e.RGB);
            }

            _model = pp.BuildModel();
            _compiledModel = OVCore.Shared.CompileModel(_model, "CPU");

            Console.WriteLine($"Model loaded successfully");
            Console.WriteLine($"Input shape: {_model.Inputs.Primary.Shape}");
            Console.WriteLine($"Output shape: {_model.Outputs.Primary.Shape}");
        }

        public void Run(int cameraIndex = 0)
        {
            using VideoCapture capture = new VideoCapture(cameraIndex);

            if (!capture.IsOpened())
            {
                Console.WriteLine($"Failed to open camera {cameraIndex}");
                return;
            }

            int frameWidth = (int)capture.Get(VideoCaptureProperties.FrameWidth);
            int frameHeight = (int)capture.Get(VideoCaptureProperties.FrameHeight);
            Console.WriteLine($"Camera opened: {frameWidth}x{frameHeight}");
            Console.WriteLine("Press 'q' or ESC to exit");

            using Mat frame = new Mat();
            using InferRequest ir = _compiledModel.CreateInferRequest();
            Shape inputShape = _model.Inputs.Primary.Shape;
            int modelWidth = inputShape[2];
            int modelHeight = inputShape[1];

            Stopwatch fpsWatch = Stopwatch.StartNew();
            int frameCount = 0;
            double fps = 0;

            while (true)
            {
                capture.Read(frame);
                if (frame.Empty())
                    break;

                frameCount++;
                if (fpsWatch.ElapsedMilliseconds >= 1000)
                {
                    fps = frameCount / (fpsWatch.ElapsedMilliseconds / 1000.0);
                    frameCount = 0;
                    fpsWatch.Restart();
                }

                // Preprocess
                Stopwatch sw = Stopwatch.StartNew();
                using Mat resized = frame.Resize(new Size(modelWidth, modelHeight));
                using Mat f32 = new Mat();
                resized.ConvertTo(f32, MatType.CV_32FC3, 1.0 / 255);

                // Inference
                using (Tensor input = f32.AsTensor())
                {
                    ir.Inputs.Primary = input;
                }

                ir.Run();

                // Post-process
                List<Detection> detections = PostProcess(ir.Outputs.Primary, frame.Width, frame.Height, modelWidth, modelHeight);

                double inferTime = sw.Elapsed.TotalMilliseconds;

                // Draw results
                DrawDetections(frame, detections);

                // Draw FPS and inference time
                string fpsText = $"FPS: {fps:F1} | Inference: {inferTime:F1}ms | Objects: {detections.Count}";
                Cv2.PutText(frame, fpsText, new Point(10, 30), HersheyFonts.HersheySimplex, 0.7, Scalar.Green, 2);

                Cv2.ImShow("YOLOv8 Object Detection - Press 'q' or ESC to exit", frame);

                int key = Cv2.WaitKey(1);
                if (key == 'q' || key == 27) // 'q' or ESC
                    break;
            }

            Cv2.DestroyAllWindows();
        }

        private List<Detection> PostProcess(Tensor output, int frameWidth, int frameHeight, int modelWidth, int modelHeight)
        {
            ReadOnlySpan<float> data = output.GetData<float>();
            Shape shape = output.Shape;

            // YOLOv8 output format: [1, 84, 8400] or [1, 4+num_classes, num_predictions]
            // We need to transpose it to [num_predictions, 84]
            int numClasses = shape[1] - 4; // 84 - 4 = 80 classes
            int numPredictions = shape[2]; // 8400

            List<Detection> allDetections = new List<Detection>();

            float scaleX = (float)frameWidth / modelWidth;
            float scaleY = (float)frameHeight / modelHeight;

            for (int i = 0; i < numPredictions; i++)
            {
                // Get class scores
                float maxScore = 0;
                int maxClassId = -1;

                for (int j = 0; j < numClasses; j++)
                {
                    int index = (4 + j) * numPredictions + i;
                    float score = data[index];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = j;
                    }
                }

                if (maxScore < _confidenceThreshold)
                    continue;

                // Get bbox coordinates (center_x, center_y, width, height)
                float cx = data[0 * numPredictions + i] * scaleX;
                float cy = data[1 * numPredictions + i] * scaleY;
                float w = data[2 * numPredictions + i] * scaleX;
                float h = data[3 * numPredictions + i] * scaleY;

                // Convert to (x1, y1, x2, y2)
                float x1 = cx - w / 2;
                float y1 = cy - h / 2;
                float x2 = cx + w / 2;
                float y2 = cy + h / 2;

                allDetections.Add(new Detection
                {
                    ClassId = maxClassId,
                    Confidence = maxScore,
                    BBox = new Rect2d(x1, y1, w, h)
                });
            }

            // Apply Non-Maximum Suppression
            return ApplyNMS(allDetections);
        }

        private List<Detection> ApplyNMS(List<Detection> detections)
        {
            if (detections.Count == 0)
                return detections;

            // Group by class
            var groupedDetections = detections.GroupBy(d => d.ClassId);
            List<Detection> result = new List<Detection>();

            foreach (var group in groupedDetections)
            {
                var classDetections = group.OrderByDescending(d => d.Confidence).ToList();
                List<Detection> keep = new List<Detection>();

                while (classDetections.Count > 0)
                {
                    var best = classDetections[0];
                    keep.Add(best);
                    classDetections.RemoveAt(0);

                    classDetections = classDetections.Where(d =>
                        CalculateIoU(best.BBox, d.BBox) < _nmsThreshold
                    ).ToList();
                }

                result.AddRange(keep);
            }

            return result;
        }

        private float CalculateIoU(Rect2d box1, Rect2d box2)
        {
            double x1 = Math.Max(box1.X, box2.X);
            double y1 = Math.Max(box1.Y, box2.Y);
            double x2 = Math.Min(box1.X + box1.Width, box2.X + box2.Width);
            double y2 = Math.Min(box1.Y + box1.Height, box2.Y + box2.Height);

            double intersectionArea = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            double box1Area = box1.Width * box1.Height;
            double box2Area = box2.Width * box2.Height;
            double unionArea = box1Area + box2Area - intersectionArea;

            return (float)(intersectionArea / unionArea);
        }

        private void DrawDetections(Mat frame, List<Detection> detections)
        {
            foreach (var detection in detections)
            {
                Scalar color = _colorMap[detection.ClassId];

                // Draw bounding box
                Rect rect = new Rect(
                    (int)detection.BBox.X,
                    (int)detection.BBox.Y,
                    (int)detection.BBox.Width,
                    (int)detection.BBox.Height
                );
                Cv2.Rectangle(frame, rect, color, 2);

                // Prepare label
                string label = $"{_labels[detection.ClassId]}: {detection.Confidence:P0}";

                // Calculate text size for background
                Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);

                // Draw label background
                Point labelPos = new Point(rect.X, rect.Y - 10);
                if (labelPos.Y < 0) labelPos.Y = rect.Y + 20;

                Cv2.Rectangle(frame,
                    new Rect(labelPos.X, labelPos.Y - textSize.Height - 5, textSize.Width, textSize.Height + 5),
                    color, -1);

                // Draw label text
                Cv2.PutText(frame, label, labelPos, HersheyFonts.HersheySimplex, 0.5, Scalar.White, 1);
            }
        }

        public void Dispose()
        {
            _compiledModel?.Dispose();
            _model?.Dispose();
        }
    }

    public class Detection
    {
        public int ClassId { get; set; }
        public float Confidence { get; set; }
        public Rect2d BBox { get; set; }
    }
}
